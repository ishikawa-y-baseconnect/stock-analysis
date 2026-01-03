"""価格レンジ予測モジュール

3-6ヶ月後の予想高値・安値を予測するMLモデル
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from pharma_stock.storage.models import StockPrice, PipelineEvent


@dataclass
class PriceRangePrediction:
    """価格レンジ予測結果"""

    ticker: str
    prediction_date: date
    current_price: float

    # 3ヶ月予測
    predicted_high_3m: float
    predicted_low_3m: float
    expected_return_3m: float  # (high - current) / current
    max_drawdown_3m: float  # (low - current) / current

    # 6ヶ月予測
    predicted_high_6m: float
    predicted_low_6m: float
    expected_return_6m: float
    max_drawdown_6m: float

    # 信頼度
    confidence: float  # 0-1

    # リスクリワード
    risk_reward_3m: float  # expected_return / abs(max_drawdown)
    risk_reward_6m: float


class PriceRangePredictor:
    """価格レンジ予測器

    過去データから将来の高値・安値を予測
    """

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model_high_3m = None
        self.model_low_3m = None
        self.model_high_6m = None
        self.model_low_6m = None
        self.is_trained = False

    def prepare_features(
        self,
        prices: list[StockPrice],
        pipeline_events: list[PipelineEvent] | None = None,
        fundamental_data: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """特徴量を準備"""

        df = self._prices_to_dataframe(prices)
        df = self._add_technical_features(df)

        # パイプライン特徴量
        if pipeline_events:
            df = self._add_pipeline_features(df, pipeline_events)

        # ファンダメンタル特徴量
        if fundamental_data:
            df = self._add_fundamental_features(df, fundamental_data)

        return df

    def prepare_training_data(
        self,
        prices: list[StockPrice],
        horizon_3m: int = 63,  # 約3ヶ月の営業日
        horizon_6m: int = 126,  # 約6ヶ月の営業日
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """学習データを準備

        Returns:
            X: 特徴量
            y: ターゲット (high_3m, low_3m, high_6m, low_6m)
        """
        df = self._prices_to_dataframe(prices)
        df = self._add_technical_features(df)

        # 将来の高値・安値を計算
        df['future_high_3m'] = df['close'].rolling(window=horizon_3m).max().shift(-horizon_3m)
        df['future_low_3m'] = df['close'].rolling(window=horizon_3m).min().shift(-horizon_3m)
        df['future_high_6m'] = df['close'].rolling(window=horizon_6m).max().shift(-horizon_6m)
        df['future_low_6m'] = df['close'].rolling(window=horizon_6m).min().shift(-horizon_6m)

        # リターンに変換
        df['return_high_3m'] = (df['future_high_3m'] - df['close']) / df['close']
        df['return_low_3m'] = (df['future_low_3m'] - df['close']) / df['close']
        df['return_high_6m'] = (df['future_high_6m'] - df['close']) / df['close']
        df['return_low_6m'] = (df['future_low_6m'] - df['close']) / df['close']

        # NaN除去
        df = df.dropna()

        # 特徴量とターゲットを分離
        feature_cols = [
            'RSI', 'BB_position', 'SMA_5_ratio', 'SMA_20_ratio', 'SMA_60_ratio',
            'volatility_20', 'volume_ratio', 'price_from_52w_high', 'price_from_52w_low',
            'momentum_5', 'momentum_20', 'consecutive_down', 'consecutive_up',
        ]

        # 存在する特徴量のみ使用
        feature_cols = [c for c in feature_cols if c in df.columns]

        X = df[feature_cols]
        y = df[['return_high_3m', 'return_low_3m', 'return_high_6m', 'return_low_6m']]

        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> dict[str, float]:
        """モデルを学習

        Returns:
            各ターゲットのMAE
        """
        metrics = {}

        # 時系列分割でクロスバリデーション
        tscv = TimeSeriesSplit(n_splits=3)

        if self.model_type == "xgboost" and HAS_XGBOOST:
            model_class = xgb.XGBRegressor
            model_params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': 42,
            }
        elif self.model_type == "lightgbm" and HAS_LIGHTGBM:
            model_class = lgb.LGBMRegressor
            model_params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': 42,
                'verbose': -1,
            }
        else:
            # フォールバック: 線形回帰
            from sklearn.linear_model import Ridge
            model_class = Ridge
            model_params = {'alpha': 1.0}

        # 各ターゲットに対してモデルを学習
        targets = {
            'high_3m': 'return_high_3m',
            'low_3m': 'return_low_3m',
            'high_6m': 'return_high_6m',
            'low_6m': 'return_low_6m',
        }

        for name, col in targets.items():
            model = model_class(**model_params)

            # クロスバリデーション
            cv_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[col].iloc[train_idx], y[col].iloc[val_idx]

                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                cv_scores.append(mean_absolute_error(y_val, pred))

            metrics[f'mae_{name}'] = np.mean(cv_scores)

            # 全データで再学習
            model.fit(X, y[col])
            setattr(self, f'model_{name}', model)

        self.is_trained = True
        self.feature_cols = list(X.columns)

        return metrics

    def predict(
        self,
        prices: list[StockPrice],
        pipeline_events: list[PipelineEvent] | None = None,
        fundamental_data: dict[str, Any] | None = None,
    ) -> PriceRangePrediction:
        """価格レンジを予測"""

        df = self.prepare_features(prices, pipeline_events, fundamental_data)

        # 最新の特徴量を取得
        latest = df.iloc[-1]
        current_price = float(latest['close'])
        latest_date = df.index[-1]
        if hasattr(latest_date, 'date'):
            latest_date = latest_date.date()

        # 特徴量を抽出
        feature_cols = [
            'RSI', 'BB_position', 'SMA_5_ratio', 'SMA_20_ratio', 'SMA_60_ratio',
            'volatility_20', 'volume_ratio', 'price_from_52w_high', 'price_from_52w_low',
            'momentum_5', 'momentum_20', 'consecutive_down', 'consecutive_up',
        ]
        feature_cols = [c for c in feature_cols if c in df.columns]

        if self.is_trained:
            X = df[feature_cols].iloc[[-1]]

            return_high_3m = float(self.model_high_3m.predict(X)[0])
            return_low_3m = float(self.model_low_3m.predict(X)[0])
            return_high_6m = float(self.model_high_6m.predict(X)[0])
            return_low_6m = float(self.model_low_6m.predict(X)[0])
            confidence = 0.7  # 学習済みモデル
        else:
            # 未学習の場合はヒューリスティックベース
            return_high_3m, return_low_3m, return_high_6m, return_low_6m, confidence = \
                self._heuristic_prediction(latest, df)

        # 予測価格を計算
        predicted_high_3m = current_price * (1 + return_high_3m)
        predicted_low_3m = current_price * (1 + return_low_3m)
        predicted_high_6m = current_price * (1 + return_high_6m)
        predicted_low_6m = current_price * (1 + return_low_6m)

        # リスクリワード計算
        rr_3m = return_high_3m / abs(return_low_3m) if return_low_3m < 0 else float('inf')
        rr_6m = return_high_6m / abs(return_low_6m) if return_low_6m < 0 else float('inf')

        return PriceRangePrediction(
            ticker=prices[0].ticker,
            prediction_date=latest_date,
            current_price=current_price,
            predicted_high_3m=predicted_high_3m,
            predicted_low_3m=predicted_low_3m,
            expected_return_3m=return_high_3m,
            max_drawdown_3m=return_low_3m,
            predicted_high_6m=predicted_high_6m,
            predicted_low_6m=predicted_low_6m,
            expected_return_6m=return_high_6m,
            max_drawdown_6m=return_low_6m,
            confidence=confidence,
            risk_reward_3m=rr_3m,
            risk_reward_6m=rr_6m,
        )

    def _heuristic_prediction(
        self,
        latest: pd.Series,
        df: pd.DataFrame,
    ) -> tuple[float, float, float, float, float]:
        """ヒューリスティックベースの予測

        学習データがない場合のフォールバック
        """
        # 過去のボラティリティから推定
        volatility = latest.get('volatility_20', 0.02)
        rsi = latest.get('RSI', 50)
        bb_position = latest.get('BB_position', 0.5)
        price_from_52w_low = latest.get('price_from_52w_low', 0)

        # ベースライン予測
        base_up_3m = volatility * np.sqrt(63) * 1.5  # 上昇余地
        base_down_3m = -volatility * np.sqrt(63) * 1.0  # 下落リスク

        # RSIによる調整
        if rsi < 30:  # 売られすぎ
            base_up_3m *= 1.5
            base_down_3m *= 0.7
        elif rsi > 70:  # 買われすぎ
            base_up_3m *= 0.7
            base_down_3m *= 1.5

        # BB位置による調整
        if bb_position < 0.2:  # 下限近く
            base_up_3m *= 1.3
            base_down_3m *= 0.8
        elif bb_position > 0.8:  # 上限近く
            base_up_3m *= 0.8
            base_down_3m *= 1.3

        # 52週安値からの位置による調整
        if price_from_52w_low < 0.1:  # 52週安値に近い
            base_up_3m *= 1.4
            base_down_3m *= 0.6

        # 6ヶ月は3ヶ月の√2倍程度
        base_up_6m = base_up_3m * 1.4
        base_down_6m = base_down_3m * 1.4

        # 上限・下限を設定
        return_high_3m = min(max(base_up_3m, 0.05), 0.8)
        return_low_3m = max(min(base_down_3m, -0.05), -0.4)
        return_high_6m = min(max(base_up_6m, 0.08), 1.2)
        return_low_6m = max(min(base_down_6m, -0.08), -0.5)

        confidence = 0.5  # ヒューリスティックは信頼度低め

        return return_high_3m, return_low_3m, return_high_6m, return_low_6m, confidence

    def _prices_to_dataframe(self, prices: list[StockPrice]) -> pd.DataFrame:
        """株価データをDataFrameに変換"""
        data = [{
            'date': p.date,
            'open': float(p.open),
            'high': float(p.high),
            'low': float(p.low),
            'close': float(p.close),
            'volume': p.volume,
        } for p in prices]

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        return df

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """テクニカル特徴量を追加"""
        import ta

        # RSI
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)

        # ボリンジャーバンド
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

        # 移動平均比率
        df['SMA_5'] = ta.trend.sma_indicator(df['close'], window=5)
        df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['SMA_60'] = ta.trend.sma_indicator(df['close'], window=60)
        df['SMA_5_ratio'] = df['close'] / df['SMA_5'] - 1
        df['SMA_20_ratio'] = df['close'] / df['SMA_20'] - 1
        df['SMA_60_ratio'] = df['close'] / df['SMA_60'] - 1

        # ボラティリティ
        df['returns'] = df['close'].pct_change()
        df['volatility_20'] = df['returns'].rolling(20).std()

        # 出来高比率
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # 52週高値・安値からの位置
        df['high_52w'] = df['high'].rolling(252).max()
        df['low_52w'] = df['low'].rolling(252).min()
        df['price_from_52w_high'] = (df['close'] - df['high_52w']) / df['high_52w']
        df['price_from_52w_low'] = (df['close'] - df['low_52w']) / df['low_52w']

        # モメンタム
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_20'] = df['close'].pct_change(20)

        # 連続下落・上昇日数
        df['is_down'] = (df['close'] < df['close'].shift(1)).astype(int)
        df['is_up'] = (df['close'] > df['close'].shift(1)).astype(int)

        consecutive_down = []
        consecutive_up = []
        down_count = 0
        up_count = 0

        for i in range(len(df)):
            if df['is_down'].iloc[i]:
                down_count += 1
                up_count = 0
            elif df['is_up'].iloc[i]:
                up_count += 1
                down_count = 0
            else:
                down_count = 0
                up_count = 0
            consecutive_down.append(down_count)
            consecutive_up.append(up_count)

        df['consecutive_down'] = consecutive_down
        df['consecutive_up'] = consecutive_up

        return df

    def _add_pipeline_features(
        self,
        df: pd.DataFrame,
        events: list[PipelineEvent],
    ) -> pd.DataFrame:
        """パイプライン特徴量を追加"""
        # Phase 3試験数
        phase3_count = sum(1 for e in events if e.phase and '3' in e.phase)
        df['phase3_count'] = phase3_count

        # 進行中の試験数
        active_count = sum(1 for e in events if e.status and 'Recruiting' in e.status)
        df['active_trials'] = active_count

        return df

    def _add_fundamental_features(
        self,
        df: pd.DataFrame,
        fundamental: dict[str, Any],
    ) -> pd.DataFrame:
        """ファンダメンタル特徴量を追加"""
        df['PER'] = fundamental.get('trailingPE', np.nan)
        df['PBR'] = fundamental.get('priceToBook', np.nan)
        df['dividend_yield'] = fundamental.get('dividendYield', 0) * 100 if fundamental.get('dividendYield') else 0

        return df


def backtest_prediction(
    predictor: PriceRangePredictor,
    prices: list[StockPrice],
    test_start_idx: int,
    horizon_3m: int = 63,
    horizon_6m: int = 126,
) -> pd.DataFrame:
    """予測のバックテスト

    Args:
        predictor: 予測器
        prices: 全期間の株価データ
        test_start_idx: テスト開始インデックス
        horizon_3m: 3ヶ月の営業日数
        horizon_6m: 6ヶ月の営業日数

    Returns:
        バックテスト結果のDataFrame
    """
    results = []

    for i in range(test_start_idx, len(prices) - horizon_6m):
        # その時点までのデータで予測
        train_prices = prices[:i+1]

        prediction = predictor.predict(train_prices)

        # 実際の高値・安値を計算
        future_prices = prices[i+1:i+1+horizon_6m]
        if len(future_prices) < horizon_3m:
            continue

        actual_high_3m = max(float(p.high) for p in future_prices[:horizon_3m])
        actual_low_3m = min(float(p.low) for p in future_prices[:horizon_3m])
        actual_high_6m = max(float(p.high) for p in future_prices[:horizon_6m])
        actual_low_6m = min(float(p.low) for p in future_prices[:horizon_6m])

        current = prediction.current_price

        results.append({
            'date': prediction.prediction_date,
            'current_price': current,
            'pred_high_3m': prediction.predicted_high_3m,
            'actual_high_3m': actual_high_3m,
            'pred_low_3m': prediction.predicted_low_3m,
            'actual_low_3m': actual_low_3m,
            'pred_return_3m': prediction.expected_return_3m,
            'actual_return_3m': (actual_high_3m - current) / current,
            'pred_high_6m': prediction.predicted_high_6m,
            'actual_high_6m': actual_high_6m,
            'pred_low_6m': prediction.predicted_low_6m,
            'actual_low_6m': actual_low_6m,
            'pred_return_6m': prediction.expected_return_6m,
            'actual_return_6m': (actual_high_6m - current) / current,
        })

    return pd.DataFrame(results)
