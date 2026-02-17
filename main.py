"""
FAR_stock - Financial AI Recommender for Stock Surge
メインStreamlitアプリケーション

5タブ構成:
1. 📌 個別銘柄分析（チャート + シグナル + ML確率）
2. 🔍 セクター別スクリーニング
3. 🚀 急騰候補AI検知（メイン機能）
4. 📊 バックテスト結果
5. ⭐ ウォッチリスト管理
"""

import os
import sys
import time
import streamlit as st
import pandas as pd
import numpy as np

# パス設定
sys.path.insert(0, os.path.dirname(__file__))

from config import ScanConfig, ModelConfig, APIConfig
from src.data_loader import get_stock_data, fetch_jpx_tickers
from src.analyzer import (
    add_technical_indicators, analyze_term_signal,
    analyze_speculative_signal
)
from src.feature_engineering import FeatureEngineer
from src.ml_model import SurgePredictor
from src.recommender import StockRecommender
from src.utils import validate_ticker_symbol, plot_stock_chart
from src import db
from src.trainer import TFTTrainer
from src.rl.agent import PortfolioOptimizer
from config import TFTConfig, RLConfig
import plotly.graph_objects as go
# ───────────────────────────────────────────
# Page Config
# ───────────────────────────────────────────
st.set_page_config(
    page_title="FAR_stock - 急騰株AI推薦",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 FAR_stock - 急騰株AIレコメンダー")
st.caption("Financial AI Recommender for Stock Surge | LightGBM + ルールベース アンサンブル推薦")


# ───────────────────────────────────────────
# Model Loading (Cached)
# ───────────────────────────────────────────
@st.cache_resource
def load_model():
    """学習済みLightGBMモデルを読み込む（存在する場合）。"""
    predictor = SurgePredictor()
    model_path = "models/lgbm_surge_v1.pkl"
    if os.path.exists(model_path):
        try:
            predictor.load(model_path)
            return predictor
        except Exception as e:
            st.warning(f"モデル読み込みエラー: {e}")
    return predictor


@st.cache_data(ttl=3600)
def load_tickers():
    """銘柄一覧をキャッシュ付きで取得する。"""
    return fetch_jpx_tickers()


model = load_model()
recommender = StockRecommender(model=model)

# ───────────────────────────────────────────
# Sidebar - Model Training & Watchlist
# ───────────────────────────────────────────
st.sidebar.header("⚙️ システム管理")

# Model Status
model_status = "✅ 学習済み" if model.model is not None else "⚠️ 未学習"
st.sidebar.markdown(f"**MLモデル状態**: {model_status}")

from config import GPU_AVAILABLE
gpu_status = "🟢 GPU (RTX)" if GPU_AVAILABLE else "🔴 CPU"
st.sidebar.markdown(f"**デバイス**: {gpu_status}")

if model.model is not None and model.metrics_:
    m = model.metrics_
    st.sidebar.markdown(f"- AUC: {m.get('auc_roc', 0):.3f}")
    st.sidebar.markdown(f"- P@{ModelConfig.TOP_K}: {m.get('precision_at_k', 0):.3f}")

# TFT Model Status
tft_status = "✅ 学習済み" if os.path.exists(TFTConfig.MODEL_DIR) and any(f.endswith('.ckpt') for f in os.listdir(TFTConfig.MODEL_DIR)) else "⚠️ 未学習"
st.sidebar.markdown(f"**TFTモデル状態**: {tft_status}")

# Train Button
if st.sidebar.button("🧠 モデル学習/再学習"):
    with st.sidebar.status("モデル学習中...", expanded=True) as status:
        st.write("銘柄一覧を取得中...")
        tickers_df = load_tickers()

        if tickers_df.empty:
            st.sidebar.error("銘柄一覧の取得に失敗")
        else:
            # 学習用データ収集: 層化サンプリング + 低位株重点
            st.write("層化サンプリング + 低位株重点でサンプリング中...")

            # 1. 各セクターから均等にサンプリング
            sectors = tickers_df['Sector'].unique()
            per_sector = max(4, 150 // len(sectors))  # 全体で約150銘柄
            stratified = []
            for sector in sectors:
                sector_df = tickers_df[tickers_df['Sector'] == sector]
                n_sample = min(per_sector, len(sector_df))
                stratified.append(sector_df.sample(n=n_sample, random_state=42))
            stratified_df = pd.concat(stratified, ignore_index=True)

            # 2. 低位株を追加サンプリング (一括ダウンロードで高速化)
            #    株価1,000円以下の銘柄を重点的に追加
            st.write("📉 **Phase 1/3**: 低位株の株価チェック中...")
            low_price_candidates = []
            check_tickers = tickers_df[
                ~tickers_df['Ticker'].isin(stratified_df['Ticker'])
            ]['Ticker'].tolist()

            import random
            import yfinance as yf
            random.seed(42)
            check_sample = random.sample(check_tickers, min(500, len(check_tickers)))

            # 一括ダウンロードで高速化 (50銘柄ずつ)
            progress_phase1 = st.progress(0, text="低位株チェック中...")
            chunk_size = 50
            for chunk_i in range(0, len(check_sample), chunk_size):
                chunk = check_sample[chunk_i:chunk_i + chunk_size]
                try:
                    data = yf.download(chunk, period="5d", progress=False, threads=True)
                    if not data.empty:
                        if isinstance(data.columns, pd.MultiIndex):
                            for t in chunk:
                                try:
                                    close = data.xs(t, axis=1, level=1)['Close'].dropna()
                                    if not close.empty and close.iloc[-1] <= ScanConfig.MAX_PRICE:
                                        low_price_candidates.append(t)
                                except (KeyError, Exception):
                                    pass
                        elif len(chunk) == 1 and 'Close' in data.columns:
                            if data['Close'].iloc[-1] <= ScanConfig.MAX_PRICE:
                                low_price_candidates.append(chunk[0])
                except Exception:
                    pass

                progress_phase1.progress(
                    min((chunk_i + chunk_size), len(check_sample)) / len(check_sample),
                    text=f"低位株チェック: {min(chunk_i+chunk_size, len(check_sample))}/{len(check_sample)} (発見: {len(low_price_candidates)}件)"
                )

                if len(low_price_candidates) >= 50:
                    break

            progress_phase1.progress(1.0, text=f"低位株チェック完了 ✅ ({len(low_price_candidates)}件)")

            # 3. 統合: 層化サンプル + 低位株追加
            all_sample = list(set(
                stratified_df['Ticker'].tolist() + low_price_candidates
            ))
            sample_tickers = all_sample

            st.write(
                f"**{len(sample_tickers)}銘柄** をサンプリング完了 "
                f"(層化: {len(stratified_df)}銘柄 × {len(sectors)}セクター, "
                f"低位株追加: {len(low_price_candidates)}銘柄)"
            )

            # 4. 特徴量生成 & データ収集 (一括ダウンロードで高速化)
            st.write("📊 **Phase 2/3**: データ一括取得 & 特徴量生成中...")
            fe = FeatureEngineer()
            all_X = []
            all_y = []
            fetched_count = 0

            # yf.download で一括取得 (50銘柄チャンク)
            progress_phase2 = st.progress(0, text="データ取得中...")
            dl_chunk_size = 50
            all_data = {}
            for chunk_i in range(0, len(sample_tickers), dl_chunk_size):
                chunk = sample_tickers[chunk_i:chunk_i + dl_chunk_size]
                try:
                    data = yf.download(chunk, period="1y", progress=False, threads=True)
                    if not data.empty:
                        if isinstance(data.columns, pd.MultiIndex):
                            for t in chunk:
                                try:
                                    df_t = data.xs(t, axis=1, level=1).dropna(how='all')
                                    if not df_t.empty and len(df_t) >= 100:
                                        all_data[t] = df_t
                                except (KeyError, Exception):
                                    pass
                        elif len(chunk) == 1 and len(data) >= 100:
                            all_data[chunk[0]] = data
                except Exception:
                    pass

                progress_phase2.progress(
                    min(chunk_i + dl_chunk_size, len(sample_tickers)) / len(sample_tickers),
                    text=f"データ取得: {min(chunk_i+dl_chunk_size, len(sample_tickers))}/{len(sample_tickers)}"
                )

            st.write(f"取得完了: {len(all_data)}銘柄。特徴量生成中...")

            # 特徴量生成 (ローカル処理、高速)
            for i, (ticker, df) in enumerate(all_data.items()):
                try:
                    features = fe.build_features(df)
                    labels = fe.create_labels(df)
                    feat_cols = fe.get_feature_columns()

                    valid = features[feat_cols].notna().all(axis=1) & labels.notna()
                    if valid.sum() > 20:
                        all_X.append(features.loc[valid, feat_cols])
                        all_y.append(labels[valid])
                        fetched_count += 1
                except Exception:
                    pass

            progress_phase2.progress(1.0, text=f"特徴量生成完了 ✅ ({fetched_count}銘柄)")

            if all_X:
                X_combined = pd.concat(all_X, ignore_index=True)
                y_combined = pd.concat(all_y, ignore_index=True)

                st.write(f"学習データ: {len(X_combined)} samples, 急騰比率: {y_combined.mean():.2%}")

                st.write("🧠 **Phase 3/3**: LightGBM学習中...")
                predictor = SurgePredictor()
                metrics = predictor.train(X_combined, y_combined, walk_forward=False)

                if 'error' not in metrics:
                    predictor.save()
                    st.write(f"✅ 学習完了!")
                    st.write(f"AUC: {metrics.get('auc_roc', 0):.3f}")
                    st.write(f"P@{ModelConfig.TOP_K}: {metrics.get('precision_at_k', 0):.3f}")
                    status.update(label="学習完了!", state="complete")

                    # キャッシュクリアしてモデルリロード
                    load_model.clear()
                else:
                    st.error(f"学習エラー: {metrics}")
                    status.update(label="学習失敗", state="error")
            else:
                st.error("学習データが不足しています")
                status.update(label="データ不足", state="error")
    
# TFT Train Button
if st.sidebar.button("📈 TFT時系列学習 (GPU推奨)"):
    with st.sidebar.status("TFTモデル学習中...", expanded=True) as status:
        st.write("データ準備中...")
        tickers_df = load_tickers()
        
        # TFTは時系列データが必要なので、主要銘柄から長期間のデータを取得
        # 全銘柄は重すぎるので、上位銘柄や注目セクターを中心に
        # ここではデモとして、Liquidityの高いTop 50銘柄を使用
        target_tickers = tickers_df.head(50)['Ticker'].tolist()
        
        st.write(f"{len(target_tickers)}銘柄のデータを取得中...")
        
        import yfinance as yf
        all_data_list = []
        
        # Batch download
        data = yf.download(target_tickers, period="2y", progress=False, threads=True)
        
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                for t in target_tickers:
                    try:
                        df_t = data.xs(t, axis=1, level=1).dropna()
                        if len(df_t) > TFTConfig.MAX_ENCODER_LENGTH + TFTConfig.MAX_PREDICTION_LENGTH:
                            df_t['ticker'] = t
                            df_t = df_t.reset_index()
                            all_data_list.append(df_t)
                    except:
                        pass
            elif len(target_tickers) == 1:
                data['ticker'] = target_tickers[0]
                data = data.reset_index()
                all_data_list.append(data)
                
        if all_data_list:
            df_combined = pd.concat(all_data_list, ignore_index=True)
            st.write(f"学習データ: {len(df_combined)} records")
            
            st.write("学習開始 (数分かかります)...")
            trainer = TFTTrainer()
            best_model, _ = trainer.train(
                df_combined,
                max_epochs=TFTConfig.MAX_EPOCHS,
                batch_size=TFTConfig.BATCH_SIZE,
                learning_rate=TFTConfig.LEARNING_RATE
            )
            
            st.success("TFT学習完了!")
            status.update(label="学習完了!", state="complete")
        else:
            st.error("データ取得失敗")
            status.update(label="データ不足", state="error")

st.sidebar.divider()

# Watchlist
st.sidebar.header("⭐ ウォッチリスト")
watchlist_df = db.get_watchlist()
if not watchlist_df.empty:
    st.sidebar.write(f"保存済み: {len(watchlist_df)}件")
    for _, row in watchlist_df.iterrows():
        col1, col2 = st.sidebar.columns([3, 1])
        col1.write(row['ticker'])
        if col2.button("🗑", key=f"rm_{row['ticker']}"):
            db.remove_from_watchlist(row['ticker'])
            st.rerun()
else:
    st.sidebar.info("ウォッチリストは空です")

# ───────────────────────────────────────────
# Tabs
# ───────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📌 個別銘柄分析",
    "🔍 セクター別スクリーニング",
    "🚀 急騰候補AI検知",
    "📈 時系列予測 (TFT)",
    "⚖️ ポートフォリオ最適化 (RL)",
    "📊 バックテスト",
    "⭐ ウォッチリスト"
])

# ═══════════════════════════════════════════
# Tab 1: 個別銘柄分析
# ═══════════════════════════════════════════
with tab1:
    all_tickers = load_tickers()

    # 検索用リスト
    ticker_list = []
    if not all_tickers.empty:
        ticker_list = [f"{row.Name} ({row.Ticker})" for row in all_tickers.itertuples()]

    col1, col2 = st.columns([1, 4])
    with col1:
        ticker_input = st.text_input(
            "Ticker Symbol (例: 7203.T)",
            value="7203.T",
            key="ticker_input_tab1"
        )

        options = ["検索ワードを入力..."] + ticker_list
        selected = st.selectbox("銘柄検索", options, index=0, key="search_tab1")
        if selected != "検索ワードを入力...":
            try:
                ticker_input = selected.split("(")[-1].replace(")", "")
            except Exception:
                pass

        period_input = st.selectbox("期間", ["3mo", "6mo", "1y", "2y"], index=2)
        analyze_btn = st.button("🔍 分析開始", key="analyze_tab1")

    if analyze_btn and ticker_input:
        if not validate_ticker_symbol(ticker_input):
            st.error("無効なティッカーコードです")
        else:
            with st.spinner(f"{ticker_input} のデータを取得中..."):
                result = recommender.analyze_single(ticker_input, period=period_input)

            if 'error' in result:
                st.error(result['error'])
            else:
                df_chart = result['df']
                signals = result['signals']
                ml_prob = result['ml_prob']

                # ウォッチリスト追加ボタン
                with col2:
                    if not db.is_in_watchlist(ticker_input):
                        if st.button("⭐ ウォッチリストに追加", key="add_wl"):
                            db.add_to_watchlist(ticker_input)
                            st.rerun()

                # チャート
                fig = plot_stock_chart(df_chart, ticker_input)
                st.plotly_chart(fig, use_container_width=True)

                # ML確率表示
                if model.model is not None:
                    prob_col1, prob_col2 = st.columns([1, 4])
                    with prob_col1:
                        st.metric("🤖 ML急騰確率", f"{ml_prob:.1%}")

                # シグナル分析
                st.subheader("📊 シグナル分析")
                c1, c2, c3, c4 = st.columns(4)

                terms_map = {
                    'Short': ('📈 短期', c1),
                    'Medium': ('📊 中期', c2),
                    'Long': ('📉 長期', c3),
                }

                for term, (label, col) in terms_map.items():
                    res = signals[term]
                    with col:
                        st.markdown(f"### {label}")
                        score = res['score']
                        if score > 0:
                            st.success(f"Score: {score}")
                        elif score < 0:
                            st.error(f"Score: {score}")
                        else:
                            st.info(f"Score: {score}")
                        st.write(f"**Reason:** {res['reason']}")

                # 仕手株判定
                spec = result['speculative']
                with c4:
                    st.markdown("### 🔥 仕手株判定")
                    if spec['is_speculative']:
                        st.warning(f"Score: {spec['score']}")
                    else:
                        st.info("特になし")
                    st.write(f"Vol比: {spec['vol_ratio']:.1f}x")

# ═══════════════════════════════════════════
# Tab 2: セクター別スクリーニング
# ═══════════════════════════════════════════
with tab2:
    st.header("🔍 セクター別スクリーニング")

    all_tickers = load_tickers()

    if all_tickers.empty:
        st.error("銘柄リストの取得に失敗しました")
    else:
        sectors = sorted(all_tickers['Sector'].unique())
        selected_sector = st.selectbox("セクター選択", sectors, index=0, key="sector_tab2")

        target_term = st.selectbox(
            "目標期間",
            ["Short(短期)", "Medium(中期)", "Long(長期)"],
            index=1, key="term_tab2"
        )

        sector_tickers = all_tickers[all_tickers['Sector'] == selected_sector]
        st.write(f"**{len(sector_tickers)}** 銘柄が '{selected_sector}' セクターにあります")

        if st.button("🔄 スクリーニング開始", key="scan_tab2"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            term_key = target_term.split('(')[0]
            total = len(sector_tickers)

            for i, row in enumerate(sector_tickers.itertuples()):
                code = row.Ticker
                name = row.Name
                status_text.text(f"Scanning {code} ({name})...")

                df = get_stock_data(code, period="1y")
                if df is not None and not df.empty and len(df) >= 50:
                    try:
                        df_tech = add_technical_indicators(df)
                        res = analyze_term_signal(df_tech, term_key)
                        results.append({
                            'Ticker': code, 'Name': name,
                            'Score': res['score'],
                            'Reason': res['reason'],
                            'Price': df['Close'].iloc[-1],
                            'RSI': df_tech['RSI'].iloc[-1] if 'RSI' in df_tech.columns else None,
                        })
                    except Exception:
                        pass

                progress_bar.progress((i + 1) / total)
                time.sleep(0.3)

            status_text.text("スキャン完了!")

            if results:
                res_df = pd.DataFrame(results).sort_values('Score', ascending=False)
                st.subheader(f"推薦銘柄ランキング ({target_term})")
                st.dataframe(res_df, use_container_width=True)
            else:
                st.info("条件に合う銘柄が見つかりませんでした")

# ═══════════════════════════════════════════
# Tab 3: 急騰候補AI検知
# ═══════════════════════════════════════════
with tab3:
    st.header("🚀 急騰候補AI検知")
    st.markdown("""
    **LightGBM + ルールベースのアンサンブル** で急騰候補を検出します。
    ML確率とテクニカルシグナルを統合したスコアでランキングします。
    """)

    if model.model is None:
        st.warning("⚠️ MLモデルが未学習です。サイドバーの「モデル学習」ボタンで学習を実行してください。ルールベースのみで動作します。")

    all_tickers = load_tickers()

    if not all_tickers.empty:
        scan_mode = st.radio(
            "スキャンモード",
            ["セクター指定", "仕手株検知（出来高急増）"],
            horizontal=True, key="mode_tab3"
        )

        if scan_mode == "セクター指定":
            sectors = sorted(all_tickers['Sector'].unique())
            selected_sector = st.selectbox("セクター", sectors, key="sector_tab3")
            scan_tickers = all_tickers[all_tickers['Sector'] == selected_sector]['Ticker'].tolist()
        else:
            sectors = sorted(all_tickers['Sector'].unique())
            selected_sector = st.selectbox("セクター (仕手株)", sectors, key="spec_sector_tab3")
            scan_tickers = all_tickers[all_tickers['Sector'] == selected_sector]['Ticker'].tolist()

        st.write(f"スキャン対象: **{len(scan_tickers)}** 銘柄")

        if st.button("🚀 AIスキャン開始", key="ai_scan_tab3"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_cb(current, total):
                progress_bar.progress(current / total)
                status_text.text(f"スキャン中... ({current}/{total})")

            if scan_mode == "セクター指定":
                results_df = recommender.scan_with_ml(scan_tickers, progress_callback=progress_cb)
            else:
                results_df = recommender.scan_speculative(scan_tickers, progress_callback=progress_cb)

            status_text.text("スキャン完了!")

            if not results_df.empty:
                st.subheader("🏆 推薦ランキング")

                # スタイリング
                st.dataframe(
                    results_df.head(30),
                    use_container_width=True,
                    column_config={
                        "ML_Prob": st.column_config.ProgressColumn(
                            "ML確率", format="%.1f%%", min_value=0, max_value=1
                        ),
                        "Price": st.column_config.NumberColumn(
                            "株価", format="¥%.0f"
                        ),
                    }
                )
            else:
                st.info("条件に合う銘柄が見つかりませんでした")

# ═══════════════════════════════════════════
# Tab 4: TFT 時系列予測
# ═══════════════════════════════════════════
with tab4:
    st.header("📈 TFT 時系列予測")
    st.markdown("Temporal Fusion Transformerによる、未来5日間の株価推移予測（予測区間付き）を表示します。")
    
    col_tft1, col_tft2 = st.columns([1, 3])
    with col_tft1:
        tft_ticker = st.text_input("Ticker Symbol", value="7203.T", key="tft_ticker")
        tft_btn = st.button("🔮 予測実行", key="tft_btn")
        
    if tft_btn:
        with st.spinner("予測中..."):
            # 直近データ取得
            df_latest = get_stock_data(tft_ticker, period="6mo")
            
            if df_latest is not None and not df_latest.empty:
                # 推論実行
                pred_df = recommender.predict_tft(tft_ticker, df_latest)
                
                if not pred_df.empty:
                    st.success("予測完了!")
                    
                    # グラフ描画
                    fig = go.Figure()
                    
                    # 実測値 (直近30日)
                    recent = df_latest.iloc[-30:]
                    fig.add_trace(go.Scatter(
                        x=recent.index, y=recent['Close'],
                        mode='lines+markers', name='実測値',
                        line=dict(color='gray')
                    ))
                    
                    # 予測値
                    fig.add_trace(go.Scatter(
                        x=pred_df['Date'], y=pred_df['Predicted_Mean'],
                        mode='lines+markers', name='予測(中央値)',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # 信頼区間 (Fan Chart style)
                    fig.add_trace(go.Scatter(
                        x=pd.concat([pred_df['Date'], pred_df['Date'][::-1]]),
                        y=pd.concat([pred_df['Upper_Bound'], pred_df['Lower_Bound'][::-1]]),
                        fill='toself',
                        fillcolor='rgba(0,100,255,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='予測区間 (10-90%)'
                    ))
                    
                    fig.update_layout(
                        title=f"{tft_ticker} - 5日間価格予測",
                        xaxis_title="Date", yaxis_title="Price",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 数値表示
                    st.write("予測詳細:")
                    st.dataframe(pred_df)
                    
                else:
                    st.error("予測に失敗しました。モデルがロードできないか、データ処理エラーです。")
            else:
                st.error("データ取得に失敗しました。")

# ═══════════════════════════════════════════
# Tab 5: ポートフォリオ最適化 (RL)
# ═══════════════════════════════════════════
with tab5:
    st.header("⚖️ ポートフォリオ最適化 (DRL)")
    st.markdown("""
    **Deep Reinforcement Learning (PPO)** を用いて、
    指定された銘柄群に対する最適な資産配分（ポートフォリオ）を提案します。
    過去期間のリスク・リターンを学習し、シャープレシオの最大化を目指します。
    """)
    
    # 銘柄選択
    rl_tickers = []
    
    # ウォッチリストから選択
    wl = db.get_watchlist()
    wl_options = []
    if not wl.empty:
        wl_options = wl['ticker'].tolist()
    
    col_rl1, col_rl2 = st.columns([1, 1])
    with col_rl1:
        selected_wl = st.multiselect("ウォッチリストから選択", wl_options, default=wl_options[:5] if wl_options else None)
        rl_tickers.extend(selected_wl)
        
    with col_rl2:
        manual_tickers = st.text_area("その他 (カンマ区切り)", value="7203.T, 9984.T, 6758.T")
        if manual_tickers:
            for t in manual_tickers.split(","):
                t = t.strip()
                if validate_ticker_symbol(t) and t not in rl_tickers:
                    rl_tickers.append(t)
    
    rl_tickers = list(set(rl_tickers))  # 重複排除
    
    if len(rl_tickers) < 2:
        st.warning("最適化には少なくとも2つの銘柄が必要です。")
    else:
        st.write(f"対象銘柄 ({len(rl_tickers)}): {', '.join(rl_tickers)}")
        
        if st.button("⚖️ 最適化実行 (学習開始)", key="rl_optimize"):
            with st.spinner("データ取得 & RLエージェント学習中... (数分かかります)"):
                # データ取得
                import yfinance as yf
                data = yf.download(rl_tickers, period="2y", progress=False, threads=True)
                
                if not data.empty and len(data) > RLConfig.LOOKBACK_WINDOW + 100:
                    # Close価格のみ抽出してDataFrame化 (MultiIndex対応)
                    price_df = pd.DataFrame()
                    if isinstance(data.columns, pd.MultiIndex):
                        for t in rl_tickers:
                            try:
                                s = data.xs(t, axis=1, level=1)['Close']
                                price_df[t] = s
                            except:
                                pass
                    elif len(rl_tickers) == 1: # これはありえないが念のため
                         price_df[rl_tickers[0]] = data['Close']
                         
                    price_df.dropna(inplace=True)
                    
                    if len(price_df.columns) < 2:
                        st.error("有効な価格データが2銘柄以上揃いませんでした。")
                    else:
                        st.write(f"学習データ期間: {price_df.index.min().date()} ~ {price_df.index.max().date()} ({len(price_df)} records)")
                        
                        # 学習実行
                        optimizer = PortfolioOptimizer()
                        # 学習ステップ数を調整 (デモ用に短くするかconfig通りか)
                        optimizer.train(price_df, timesteps=10000) # デモ用に短縮
                        
                        st.success("学習完了! 最適配分を算出中...")
                        
                        # 推論 (直近データに基づく最適配分)
                        weights = optimizer.predict(price_df)
                        
                        # 結果表示
                        w_df = pd.DataFrame(list(weights.items()), columns=['Ticker', 'Weight'])
                        w_df = w_df[w_df['Weight'] > 0.01].sort_values('Weight', ascending=False) # 1%以下は省略
                        
                        c1, c2 = st.columns([1, 1])
                        with c1:
                            st.dataframe(w_df.style.format({'Weight': '{:.1%}'}))
                        
                        with c2:
                            # 円グラフ
                            fig = go.Figure(data=[go.Pie(labels=w_df['Ticker'], values=w_df['Weight'], hole=.3)])
                            fig.update_layout(title_text="推奨ポートフォリオ配分")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        # 効率的フロンティア（イメージ）やバックテストへの誘導など
                        st.info("💡 この配分に基づき、バックテストタブで検証を行うことを推奨します。")
                        
                else:
                    st.error("データ取得に失敗したか、期間が短すぎます。")

# ═══════════════════════════════════════════
# Tab 6: バックテスト
# ═══════════════════════════════════════════
with tab6:
    st.header("📊 バックテスト")
    st.markdown("推薦システムの過去パフォーマンスを検証します。")

    from src.backtester import ROIBacktester

    col1, col2, col3 = st.columns(3)
    with col1:
        bt_capital = st.number_input("初期資金 (¥)", value=1_000_000, step=100_000)
    with col2:
        bt_positions = st.slider("最大保有銘柄数", 1, 10, 5)
    with col3:
        bt_holding = st.slider("保有日数", 1, 20, 5)

    if st.button("📊 バックテスト実行", key="bt_tab4"):
        st.info("バックテスト機能は、モデル学習後にスキャン結果を用いて実行します。")
        st.markdown("""
        **使い方**:
        1. サイドバーからMLモデルを学習
        2. 「急騰候補AI検知」タブでスキャンを実行
        3. 過去の推薦結果に基づいてバックテストを実行

        **評価指標**:
        - 累積ROI (投資収益率)
        - シャープレシオ
        - 最大ドローダウン
        - 勝率
        """)

        # デモ用のメトリクス表示
        bt = ROIBacktester(
            initial_capital=bt_capital,
            max_positions=bt_positions,
            holding_days=bt_holding
        )

        if model.model is not None and model.metrics_:
            st.subheader("📈 モデル評価指標")
            m = model.metrics_
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("AUC-ROC", f"{m.get('auc_roc', 0):.3f}")
            mc2.metric(f"Precision@{ModelConfig.TOP_K}", f"{m.get('precision_at_k', 0):.3f}")
            mc3.metric("F1-Score", f"{m.get('f1', 0):.3f}")
            mc4.metric("急騰サンプル数", f"{m.get('support_positive', 0)}")

        # 特徴量重要度
        if model.model is not None:
            fi = model.get_feature_importance()
            if not fi.empty:
                st.subheader("🔑 特徴量重要度 Top 20")
                import plotly.express as px
                fig = px.bar(
                    fi.head(20),
                    x='importance', y='feature',
                    orientation='h',
                    title='Feature Importance (Gain)',
                    labels={'importance': '重要度', 'feature': '特徴量'}
                )
                fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
                st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════
# Tab 7: ウォッチリスト
# ═══════════════════════════════════════════
with tab7:
    st.header("⭐ ウォッチリスト管理")

    # 追加フォーム
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        new_ticker = st.text_input("ティッカーコード", placeholder="例: 7203.T", key="add_wl_input")
    with col2:
        new_note = st.text_input("メモ", placeholder="注目理由", key="add_wl_note")
    with col3:
        st.write("")  # spacer
        st.write("")
        if st.button("➕ 追加", key="add_wl_btn"):
            if new_ticker:
                db.add_to_watchlist(new_ticker, new_note)
                st.rerun()

    st.divider()

    # 一覧表示
    wl = db.get_watchlist()
    if not wl.empty:
        st.dataframe(wl, use_container_width=True)

        # 一括分析
        if st.button("🔍 ウォッチリスト銘柄を一括分析"):
            for _, row in wl.iterrows():
                ticker = row['ticker']
                with st.expander(f"📌 {ticker}"):
                    df = get_stock_data(ticker, period="6mo")
                    if df is not None and not df.empty:
                        df_tech = add_technical_indicators(df)
                        fig = plot_stock_chart(df_tech, ticker)
                        st.plotly_chart(fig, use_container_width=True)

                        # シグナル
                        c1, c2, c3 = st.columns(3)
                        for term, col in zip(['Short', 'Medium', 'Long'], [c1, c2, c3]):
                            res = analyze_term_signal(df_tech, term)
                            with col:
                                st.metric(term, res['score'])
                    else:
                        st.error("データ取得失敗")
    else:
        st.info("ウォッチリストに銘柄を追加してください")
