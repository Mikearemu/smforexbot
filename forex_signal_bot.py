import asyncio
import logging
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
from telegram import Bot, Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes
from dataclasses import dataclass
import schedule
import time
import threading

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

@dataclass
class Signal:
    pair: str
    action: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    timestamp: datetime
    reason: str

class ForexDataProvider:
    def __init__(self):
        self.pairs = [
            'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X',
            'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X', 'EURGBP=X',
            'EURJPY=X', 'GBPJPY=X'
        ]

    def get_historical_data(self, pair: str, period: str = "5d", interval: str = "1h") -> pd.DataFrame:
        try:
            ticker = yf.Ticker(pair)
            data = ticker.history(period=period, interval=interval)
            if data.empty:
                logger.warning(f"No data for {pair}")
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {pair}: {e}")
            return pd.DataFrame()

class TechnicalAnalyzer:
    def __init__(self):
        self.data_provider = ForexDataProvider()

    def calculate_indicators(self, data: pd.DataFrame) -> dict:
        if data.empty or len(data) < 50:
            return {}

        close = data['Close']
        high = data['High']
        low = data['Low']

        indicators = {
            'sma_20': ta.sma(close, length=20),
            'sma_50': ta.sma(close, length=50),
            'ema_12': ta.ema(close, length=12),
            'ema_26': ta.ema(close, length=26),
            'macd': ta.macd(close),
            'rsi': ta.rsi(close, length=14),
            'bb': ta.bbands(close),
            'stoch': ta.stoch(high, low, close),
            'adx': ta.adx(high, low, close, length=14)
        }

        return indicators

    def generate_signal(self, pair: str) -> Signal:
        data = self.data_provider.get_historical_data(pair)
        if data.empty:
            return None

        indicators = self.calculate_indicators(data)
        if not indicators:
            return None

        current_price = data['Close'].iloc[-1]
        signals, confidence = [], 0.0

        confidence_weights = {
            'ma_crossover': 0.3,
            'rsi': 0.25,
            'macd': 0.2,
            'bollinger': 0.15,
            'stochastic': 0.1
        }

        sma_20 = indicators['sma_20'].iloc[-1]
        sma_50 = indicators['sma_50'].iloc[-1]

        if sma_20 > sma_50:
            signals.append('BUY'); confidence += confidence_weights['ma_crossover']
        elif sma_20 < sma_50:
            signals.append('SELL'); confidence += confidence_weights['ma_crossover']

        rsi = indicators['rsi'].iloc[-1]
        if rsi < 30:
            signals.append('BUY'); confidence += confidence_weights['rsi']
        elif rsi > 70:
            signals.append('SELL'); confidence += confidence_weights['rsi']

        macd_line = indicators['macd']['MACD_12_26_9'].iloc[-1]
        macd_signal = indicators['macd']['MACDs_12_26_9'].iloc[-1]
        if macd_line > macd_signal:
            signals.append('BUY'); confidence += confidence_weights['macd']
        elif macd_line < macd_signal:
            signals.append('SELL'); confidence += confidence_weights['macd']

        bb_lower = indicators['bb']['BBL_20_2.0'].iloc[-1]
        bb_upper = indicators['bb']['BBU_20_2.0'].iloc[-1]
        if current_price <= bb_lower:
            signals.append('BUY'); confidence += confidence_weights['bollinger']
        elif current_price >= bb_upper:
            signals.append('SELL'); confidence += confidence_weights['bollinger']

        stoch_k = indicators['stoch']['STOCHk_14_3_3'].iloc[-1]
        if stoch_k < 20:
            signals.append('BUY'); confidence += confidence_weights['stochastic']
        elif stoch_k > 80:
            signals.append('SELL'); confidence += confidence_weights['stochastic']

        action = max(set(signals), key=signals.count) if signals else 'HOLD'
        reason = f"Signals: {signals}" if signals else "No clear signals"

        atr = self.calculate_atr(data)
        sl = current_price - 2 * atr if action == 'BUY' else current_price + 2 * atr
        tp = current_price + 3 * atr if action == 'BUY' else current_price - 3 * atr

        return Signal(
            pair=pair.replace('=X', ''),
            action=action,
            entry_price=current_price,
            stop_loss=sl,
            take_profit=tp,
            confidence=min(confidence, 1.0),
            timestamp=datetime.now(),
            reason=reason
        )

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        atr = ta.atr(data['High'], data['Low'], data['Close'], length=period)
        return atr.iloc[-1] if not atr.empty else 0.02 * np.std(data['Close'])

class DatabaseManager:
    def __init__(self, db_path="forex_signals.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT, action TEXT, entry_price REAL, stop_loss REAL,
                take_profit REAL, confidence REAL, timestamp TEXT, reason TEXT
            )''')
            conn.commit()

    def save_signal(self, signal: Signal):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''INSERT INTO signals
                (pair, action, entry_price, stop_loss, take_profit, confidence, timestamp, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (signal.pair, signal.action, signal.entry_price, signal.stop_loss,
                 signal.take_profit, signal.confidence, signal.timestamp.isoformat(), signal.reason))
            conn.commit()

    def get_recent_signals(self, hours: int = 24):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            cutoff = datetime.now() - timedelta(hours=hours)
            c.execute('''SELECT * FROM signals WHERE timestamp > ? ORDER BY timestamp DESC''', (cutoff.isoformat(),))
            return c.fetchall()

class ForexSignalBot:
    def __init__(self, token):
        self.token = token
        self.analyzer = TechnicalAnalyzer()
        self.db = DatabaseManager()
        self.application = None

    async def start(self):
        self.application = Application.builder().token(self.token).build()
        self.application.add_handler(CommandHandler("start", self.cmd_start))
        self.application.add_handler(CommandHandler("signals", self.cmd_signals))
        self.application.add_handler(CommandHandler("pair", self.cmd_pair))
        self.application.add_handler(CommandHandler("stats", self.cmd_stats))
        threading.Thread(target=self.run_scheduler, daemon=True).start()
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        await asyncio.Event().wait()

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [
            ["/signals", "/stats"],
            ["/pair EURUSD", "/pair GBPUSD", "/pair USDJPY"],
            ["/pair USDCHF", "/pair AUDUSD", "/pair USDCAD"],
            ["/pair NZDUSD", "/pair EURGBP", "/pair EURJPY"],
            ["/pair GBPJPY"]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        msg = """
🤖 *Forex Signal Bot* — Welcome!

Use the quick buttons below or type a command:
        """
        await update.message.reply_text(msg, parse_mode='Markdown', reply_markup=reply_markup)

    async def cmd_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🔄 Analyzing market data... Please wait.")
        for pair in self.analyzer.data_provider.pairs:
            signal = self.analyzer.generate_signal(pair)
            if signal:
                self.db.save_signal(signal)
                emoji = "🟢" if signal.action == 'BUY' else "🔴" if signal.action == 'SELL' else "⚪"
                confidence_icon = "🔥" if signal.confidence > 0.8 else "⚡" if signal.confidence > 0.6 else "💡"
                await update.message.reply_text(
                    f"{emoji} {confidence_icon} *{signal.pair}* — {signal.action}\n"
                    f"💰 Entry: `{signal.entry_price:.5f}`\n"
                    f"🛑 SL: `{signal.stop_loss:.5f}` 🎯 TP: `{signal.take_profit:.5f}`\n"
                    f"📊 Confidence: `{signal.confidence*100:.1f}%`\n"
                    f"💭 {signal.reason}\n"
                    f"⏰ {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                    , parse_mode='Markdown')

    async def cmd_pair(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text("Please provide a pair, e.g. /pair EURUSD")
            return

        pair = context.args[0].upper() + "=X"
        if pair not in self.analyzer.data_provider.pairs:
            await update.message.reply_text(f"Unsupported pair: {pair}")
            return

        signal = self.analyzer.generate_signal(pair)
        if not signal:
            await update.message.reply_text("No signal found.")
            return

        emoji = "🟢" if signal.action == 'BUY' else "🔴" if signal.action == 'SELL' else "⚪"
        confidence_icon = "🔥" if signal.confidence > 0.8 else "⚡" if signal.confidence > 0.6 else "💡"
        await update.message.reply_text(
            f"{emoji} {confidence_icon} *{signal.pair}* — {signal.action}\n"
            f"💰 Entry: `{signal.entry_price:.5f}`\n"
            f"🛑 SL: `{signal.stop_loss:.5f}` 🎯 TP: `{signal.take_profit:.5f}`\n"
            f"📊 Confidence: `{signal.confidence*100:.1f}%`\n"
            f"💭 {signal.reason}\n"
            f"⏰ {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            , parse_mode='Markdown')

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🔄 Gathering statistics... Please wait.")
        records = self.db.get_recent_signals()
        if not records:
            await update.message.reply_text("No signals recorded in the last 24 hours.")
            return

        buy = sum(1 for r in records if r[2] == 'BUY')
        sell = sum(1 for r in records if r[2] == 'SELL')
        hold = sum(1 for r in records if r[2] == 'HOLD')
        avg_conf = sum(r[6] for r in records) / len(records)

        await update.message.reply_text(
            f"📊 *24h Signal Stats*\n"
            f"🟢 Buy: {buy}, 🔴 Sell: {sell}, ⚪ Hold: {hold}\n"
            f"📉 Avg Confidence: {avg_conf*100:.1f}%\n"
            f"⏱ Total Signals: {len(records)}"
            , parse_mode='Markdown')

    def run_scheduler(self):
        schedule.every(15).minutes.do(self.check_signals)
        while True:
            schedule.run_pending()
            time.sleep(60)

    def check_signals(self):
        for pair in self.analyzer.data_provider.pairs:
            signal = self.analyzer.generate_signal(pair)
            if signal:
                self.db.save_signal(signal)

if __name__ == "__main__":
    TELEGRAM_BOT_TOKEN = "8107869895:AAGaZPT8OvROIXie-WRMQuU2Fjy53WB8bVc"
    asyncio.run(ForexSignalBot(TELEGRAM_BOT_TOKEN).start())
