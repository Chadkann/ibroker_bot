import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import threading
import time
from datetime import datetime, timedelta
from ibapi.wrapper import EWrapper
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.order import Order
import configparser
import logging


class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []
        self.raw_df = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        self.minute_df = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume','signal'])
        self.processed_df = pd.DataFrame()
        self.lock = threading.Lock()
        self.next_valid_order_id = None
        self.max_shift = 25  # Number of previous bars to consider
        self.feature_columns = ['open_norm', 'low_norm', 'close_norm', 'volume']
        for i in range(1, self.max_shift + 1):
            self.feature_columns.extend([f'previous_{i}_{col}' for col in ['open', 'high', 'low', 'close', 'volume']])
        self.model = self.load_model()
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.quantity = self.config.getint('TRADING', 'quantity')
        self.profit_percentage = self.config.getfloat('TRADING', 'profit_percentage')
        self.loss_percentage = self.config.getfloat('TRADING', 'loss_percentage')
        self.signal_cooldown = self.config.getint('TRADING', 'signal_cooldown')
        self.trades = []
        self.daily_pnl = 0
        self.total_pnl = 0
        self.win_count = 0
        self.loss_count = 0
        self.contract = self.create_contract()
        self.positions = {}
        self.last_signal_time = None
        self.current_market_price = None
        self.schedule_next_minute_aggregation()
        self.last_aggregated_minute = None
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Add configuration
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.mp = 0  
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
    
    def error(self, reqId, errorCode, errorString):
        self.logger.error(f"Error {errorCode}: {errorString}")
    
    def connectionClosed(self):
        self.logger.warning("Connection closed. Attempting to reconnect...")
        self.connect(self.config['CONNECTION']['ip'], int(self.config['CONNECTION']['port']), 1)

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        self.logger.info(f"Order {orderId} status: {status}, Filled: {filled}, Remaining: {remaining}, Avg Fill Price: {avgFillPrice}")
        if status == "Filled":
            self.logger.info(f"Order {orderId} has been completely filled.")
        elif status == "Partially Filled":
            self.logger.info(f"Order {orderId} partially filled. {filled} executed, {remaining} remaining.")
        elif status == "Cancelled":
            self.logger.info(f"Order {orderId} has been cancelled.")
        
    def realtimeBar(self, reqId, time, open_, high, low, close, volume, wap, count):
        bar_time = pd.to_datetime(datetime.fromtimestamp(time))
        print(f"{bar_time}, O: {open_}, H: {high}, L: {low}, C: {close}, V: {volume}")
        bar_data = {'high': high, 'low': low, 'close': close}
        self.check_stop_loss_take_profit(bar_data)

        with self.lock:
            new_row = pd.DataFrame({
                'time': [bar_time],
                'open': [open_],
                'high': [high],
                'low': [low],
                'close': [close],
                'volume': [volume]
            })
            self.raw_df = pd.concat([self.raw_df, new_row], ignore_index=True)
            self.raw_df.sort_values('time', inplace=True)
            # Keep only the last 25 rows
            self.raw_df = self.raw_df.tail(25)

        self.aggregate_minute_bars(bar_time)
    def tickPrice(self, reqId, tickType, price, attrib):
        if tickType == 4:  # Last price
            with self.lock:
                self.current_market_price = price
    
    def schedule_next_minute_aggregation(self):
        #Scheduling minute aggregation for top of minute
        now = datetime.now()
        next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
        delay = (next_minute - now).total_seconds()
        threading.Timer(delay, self.on_minute).start()

    def on_minute(self):
        self.aggregate_minute_bars(datetime.now())
        self.schedule_next_minute_aggregation()

    def aggregate_minute_bars(self, current_time):
        # Round down to the current minute
        current_minute = current_time.replace(second=0, microsecond=0)

        #self.logger.info(f"Aggregating for current_minute: {current_minute}")

        # Check if we've already aggregated this minute
        if hasattr(self, 'last_aggregated_minute') and self.last_aggregated_minute == current_minute:
            #self.logger.info(f"Minute {current_minute} already aggregated, skipping.")
            return

        # Get all data points from the last aggregated minute (or the beginning) up to the current minute
        if hasattr(self, 'last_aggregated_minute') and self.last_aggregated_minute is not None:
            start_time = self.last_aggregated_minute
        else:
            start_time = self.raw_df['time'].min() if not self.raw_df.empty else current_minute - timedelta(minutes=1)

        #self.logger.info(f"Using start_time: {start_time}")

        if start_time is not None:
            minute_data = self.raw_df[(self.raw_df['time'] > start_time) & (self.raw_df['time'] <= current_minute)]
        else:
            #self.logger.warning("start_time is None, using all available data")
            minute_data = self.raw_df[self.raw_df['time'] <= current_minute]

        if not minute_data.empty:
            minute_bar = pd.DataFrame({
                'time': [current_minute],
                'open': [minute_data.iloc[0]['open']],
                'high': [minute_data['high'].max()],
                'low': [minute_data['low'].min()],
                'close': [minute_data.iloc[-1]['close']],
                'volume': [minute_data['volume'].sum()],
                'signal': 0
            })

            self.minute_df = pd.concat([self.minute_df, minute_bar], ignore_index=True)
            print(f"New minute bar: {current_minute}, O: {minute_bar['open'].values[0]}, "
                     f"H: {minute_bar['high'].values[0]}, L: {minute_bar['low'].values[0]}, "
                     f"C: {minute_bar['close'].values[0]}, V: {minute_bar['volume'].values[0]}")

            # Update the last aggregated minute
            self.last_aggregated_minute = current_minute
        #else:
            #self.logger.warning(f"No data for minute {current_minute}")

        self.process_data()

    def process_data(self):
        # If enough data has been collected for algorithm
        if len(self.minute_df) >= self.max_shift + 1:
            features = self.prepare_data_for_model()
            if features is not None:
                signal = self.generate_signal(features)
                self.handle_signal(signal)

    def load_model(self):
        return joblib.load('rfc_1min_25bar_priceaction.joblib')

    def prepare_data_for_model(self):
        #Feature Engineering redacted

        # Check if all feature columns are present
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            self.logger.warning(f"Missing feature columns: {missing_features}")
            return None
        
        #self.processed_df = df

        # Return the most recent row of features as a DataFrame
        return self.processed_df.iloc[[-1]]

    def generate_signal(self, features):
        if features is None or features.empty:
            return 0  # No signal due to insufficient data

        # Generate prediction
        signal = self.model.predict(features)[0]
        self.minute_df.loc[self.minute_df.index[-1], 'signal'] = signal
        self.logger.info(f"Generated signal: {signal}")
        self.logger.info(f"Current position: {self.mp}")
        return signal
    def handle_signal(self, signal):
        self.logger.info(f"Entering handle_signal with signal: {signal}")
        current_time = datetime.now()

        if self.last_signal_time is None:
             #self.logger.info("First signal being handled")
            pass
        else:
             time_since_last_signal = (current_time - self.last_signal_time).total_seconds()
             self.logger.info(f"Time since last signal: {time_since_last_signal} seconds")
             self.logger.info(f"Signal cooldown: {self.signal_cooldown} seconds")
    
        if self.last_signal_time is None or (current_time - self.last_signal_time).total_seconds() > self.signal_cooldown:
             if signal != 0:
                 self.logger.info(f"Attempting to place order for signal: {signal}")
                 try:
                     self.place_order_if_conditions_met(signal)
                     self.last_signal_time = current_time
                     self.logger.info(f"Order placed successfully. Last signal time updated to {current_time}")
                 except Exception as e:
                     self.logger.error(f"Error placing order: {e}")
             else:
                 self.logger.info("Signal is 0, no action taken")
        else:
             self.logger.info("Signal ignored due to cooldown period")

        self.logger.info("Exiting handle_signal")

    def on_new_minute_bar(self):
        # This method should be called whenever a new minute bar is added
        signal = self.generate_signal()
        if signal != 0:  # Assuming 0 means no action
            self.place_order_if_conditions_met()

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.next_valid_order_id = orderId
        self.logger.info(f"Next Valid Order ID: {orderId}")


    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        self.positions[contract.symbol] = {
            'quantity': position,
            'avgCost': avgCost,
            'account': account
        }
    def positionEnd(self):
        print("Positions updated")
        self.positions_updated = True

    def get_positions(self):
        self.positions_updated = False
        self.reqPositions()
        timeout = 10  # 10 seconds timeout
        start_time = time.time()
        while not self.positions_updated and time.time() - start_time < timeout:
            time.sleep(0.1)
        if not self.positions_updated:
            print("Position update timed out")
        return self.positions
    def verify_position(self):
        try:
            positions = self.get_positions()
            ibkr_position = positions.get(self.contract.symbol, {}).get('quantity', 0)
            if ibkr_position != self.mp:
                self.logger.warning(f"Position mismatch: IBKR reports {ibkr_position}, but self.mp was {self.mp}")
                self.mp = ibkr_position  # Synchronize self.mp with IBKR position
            else:
                print(f"Position verified: IBKR and self.mp both report {self.mp}")
        except Exception as e:
            self.logger.error(f"Failed to verify position: {e}")

    def place_bracket_order(self, quantity, side, sltp):
        if self.next_valid_order_id is None:
            print("Error: Next valid order ID is not set.")
            return
        
        with self.lock:
            market_price = self.raw_df.close.iloc[-1]

        # Entry price, take profit, and stop loss
        entry_price = round(market_price/0.25) * 0.25 # Futures prices only accept orders on incramnets of $0.25
        if side =="Buy":
            take_profit = round((market_price * (1 + self.profit_percentage))/0.25)*0.25
            stop_loss = round((market_price * (1 - self.loss_percentage))/0.25)*0.25
            self.stop_loss = stop_loss - .5
            self.take_profit = take_profit + .5
            stop_loss -= 1
            take_profit += 1
        elif side =="Sell":
            take_profit = round((market_price * (1 - self.profit_percentage))/0.25)*0.25
            stop_loss = round((market_price * (1 + self.loss_percentage))/0.25)*0.25
            self.stop_loss = stop_loss + .5
            self.take_profit = take_profit - .5
            stop_loss += 1
            take_profit -= 1
        else:
            self.logger.error(f"Invalid side: {side}. Must be 'Buy' or 'Sell'.")
            return
        # Main entry order
        entry = Order()
        entry.orderId = self.next_valid_order_id
        entry.action = side
        entry.orderType = "MKT"
        entry.totalQuantity = quantity
        entry.lmtPrice = entry_price
        entry.transmit = False
        entry.eTradeOnly = False
        entry.firmQuoteOnly = False

        # Stop Loss order
        stop_loss_order = Order()
        stop_loss_order.orderId = self.next_valid_order_id + 1
        stop_loss_order.action = sltp
        stop_loss_order.orderType = "STP"
        stop_loss_order.totalQuantity = quantity
        stop_loss_order.auxPrice = stop_loss
        stop_loss_order.parentId = entry.orderId
        stop_loss_order.transmit = False
        stop_loss_order.eTradeOnly = False
        stop_loss_order.firmQuoteOnly = False
        stop_loss_order.triggerMethod = 2
        stop_loss_order.outsideRth = True

        # Take Profit order
        take_profit_order = Order()
        take_profit_order.orderId = self.next_valid_order_id + 2
        take_profit_order.action = sltp
        take_profit_order.orderType = "LMT"
        take_profit_order.totalQuantity = quantity
        take_profit_order.lmtPrice = take_profit
        take_profit_order.parentId = entry.orderId
        take_profit_order.transmit = True
        take_profit_order.eTradeOnly = False
        take_profit_order.firmQuoteOnly = False

        # Place orders
        self.placeOrder(entry.orderId, self.contract, entry)
        self.placeOrder(stop_loss_order.orderId, self.contract, stop_loss_order)
        self.placeOrder(take_profit_order.orderId, self.contract, take_profit_order)

        self.logger.info(f"Placed bracket order: Entry at {entry_price}, Stop Loss at {stop_loss}, Take Profit at {take_profit}")
        # Update next valid order ID
        self.next_valid_order_id += 3
    
    def cancel_all_orders(self):
        self.logger.info("Attempting to cancel all open orders")

        # Request all open orders
        self.open_orders = []
        self.open_orders_req_completed = False
        self.reqAllOpenOrders()

        # Wait for the open orders request to complete
        timeout = time.time() + 10  # 10 seconds timeout
        while not self.open_orders_req_completed and time.time() < timeout:
            time.sleep(0.1)

        if not self.open_orders_req_completed:
            self.logger.warning("Open orders request timed out")
            return

        # Cancel each open order
        for order_id in self.open_orders:
            self.cancelOrder(order_id)
            self.logger.info(f"Cancellation request sent for order ID: {order_id}")

        # Wait for all cancellations to be processed
        cancellation_timeout = time.time() + 30  # 30 seconds timeout
        while self.open_orders and time.time() < cancellation_timeout:
            time.sleep(0.5)
            # Refresh open orders
            self.open_orders = []
            self.open_orders_req_completed = False
            self.reqAllOpenOrders()
            while not self.open_orders_req_completed and time.time() < cancellation_timeout:
                time.sleep(0.1)

        if self.open_orders:
            self.logger.warning(f"Not all orders were cancelled. Remaining orders: {self.open_orders}")
        else:
            self.logger.info("All orders have been successfully cancelled")
    
  

    def close_position(self, current_position):
        order = Order()
        if current_position < 0:
            order.action = 'Buy'
        elif current_position > 0:
            order.action = 'Sell' 
        order.totalQuantity = abs(current_position)
        order.orderType = "MKT"
        order.eTradeOnly = False
        order.firmQuoteOnly = False
        

   
        self.placeOrder(self.next_valid_order_id, self.contract, order)
        self.next_valid_order_id += 1
        self.mp=0

        self.logger.info(f"Closing position: {order.action} {order.totalQuantity} {self.contract.symbol}")

    def place_order_if_conditions_met(self,signal):
        try:
            self.verify_position()
            current_position = self.mp  # Use the verified and updated self.mp
            print(f"Verified current position: {current_position}")
        except Exception as e:
            self.logger.error(f"Failed to verify position with IBKR: {e}")
            print(f"Using current self.mp value: {self.mp}")
            current_position = self.mp
        if current_position==0:
            print('No current position, entering new trade')
            if signal == 1:
                self.place_bracket_order(quantity=self.quantity, 
                                         side="Buy", sltp='Sell')
                self.mp += 1
            elif signal == 2:
                self.place_bracket_order(quantity=self.quantity, 
                                         side="Sell",sltp='Buy')
                self.mp -= 1
        if current_position > 0:
            if signal == 2:
                self.reqGlobalCancel()
                self.close_position(current_position)
                self.place_bracket_order(quantity=self.quantity,
                                     side="Sell",sltp='Buy')
                self.mp = -1
            else:
                print('Already in Long position')
        if current_position < 0:
            if signal == 1:
                self.reqGlobalCancel()
                self.close_position(current_position)
                self.place_bracket_order(quantity=self.quantity,
                                     side="Buy", sltp='Sell')
                self.mp = 1 
            else:  
                print('Already in Short position')

    def check_stop_loss_take_profit(self, bar_data):
        if self.mp == 0 or self.stop_loss is None or self.take_profit is None:
            return  # No position or levels not set

        if self.mp > 0:  # Long position
            if bar_data['low'] <= self.stop_loss:
                print(f"Stop-loss triggered for long position. Low: {bar_data['low']}, Stop-loss: {self.stop_loss}")
                self.reqGlobalCancel()
                self.close_position(self.mp)
            elif bar_data['high'] >= self.take_profit:
                print(f"Take-profit triggered for long position. High: {bar_data['high']}, Take-profit: {self.take_profit}")
                self.reqGlobalCancel()
                self.close_position(self.mp)
        elif self.mp < 0:  # Short position
            if bar_data['high'] >= self.stop_loss:
                print(f"Stop-loss triggered for short position. High: {bar_data['high']}, Stop-loss: {self.stop_loss}")
                self.reqGlobalCancel()
                self.close_position(self.mp)
            elif bar_data['low'] <= self.take_profit:
                print(f"Take-profit triggered for short position. Low: {bar_data['low']}, Take-profit: {self.take_profit}")
                self.reqGlobalCancel()
                self.close_position(self.mp)
                
    def create_contract(self):
        contract = Contract()
        contract.symbol = self.config['CONTRACT']['symbol']
        contract.secType = self.config['CONTRACT']['secType']
        contract.exchange =self.config['CONTRACT']['exchange']
        contract.currency =self.config['CONTRACT']['currency']
        contract.lastTradeDateOrContractMonth = self.config['CONTRACT']['expiry']
        return contract
    

    def on_trade_completed(self, trade_data):
        # Call this method when a trade is completed
        self.trades.append(trade_data)
        if trade_data['realized_pnl'] > 0:
            self.win_count += 1
        else:
            self.loss_count += 1

    def generatePerformanceReport(self):
        end_balance = self.get_account_balance()
        total_return = (end_balance - self.start_balance) / self.start_balance * 100
        win_rate = self.win_count / (self.win_count + self.loss_count) * 100 if (self.win_count + self.loss_count) > 0 else 0

        report = f"""
        Performance Report
        ==================
        Start Balance: ${self.start_balance:.2f}
        End Balance: ${end_balance:.2f}
        Total Return: {total_return:.2f}%
        Total PnL: ${self.total_pnl:.2f}
        Number of Trades: {len(self.trades)}
        Win Rate: {win_rate:.2f}%

        Daily PnL: ${self.daily_pnl:.2f}

        Last 10 Trades:
        """

        for trade in self.trades[-10:]:
            report += f"Date: {trade['date']}, PnL: ${trade['realized_pnl']:.2f}\n"

        self.logger.info(report)

        # Optionally, save the report to a file
        with open(f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
            f.write(report)



# Load in Config for ports, current contract
config = configparser.ConfigParser()
config.read('config.ini')

app = IBapi()
app.connect(config['CONNECTION']['ip'], int(config['CONNECTION']['port']),0)

def run_loop():
    app.run()

api_thread = threading.Thread(target=run_loop, daemon=True)
api_thread.start()
time.sleep(1) 

contract = app.create_contract() 

# Request real-time 5-second bars
app.reqRealTimeBars(1, contract, 5, "TRADES", 0, [])
#app.reqAccountUpdates(True, "")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
try:
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("Data collection interrupted by user")
    app.minute_df.to_csv(f'{timestamp}_1m.csv')
    app.cancelRealTimeBars(1)
    if hasattr(app, 'position') and isinstance(app.position, dict):
        app.reqGlobalCancel()
        app.close_position(app.position)
    else:
        print("No valid position to close")
    
    app.disconnect()
