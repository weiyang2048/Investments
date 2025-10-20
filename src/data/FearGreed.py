import fear_and_greed
import pytz
import requests
import datetime


class FearGreed:
    def __init__(self, symbols: list = None):
        pass

    def get_fear_and_greed(self) -> dict:
        """Get fear and greed index data and return as dictionary."""
        fng_data = fear_and_greed.get()
        est = pytz.timezone("US/Eastern")
        last_update_est = fng_data.last_update.astimezone(est)
        value = int(round(fng_data.value, 0))
        description = fng_data.description
        last_update_est_str = last_update_est.strftime("%H:%M:%S  %Y-%m-%d %Z")
        self.fear_and_greed = {"value": value, "description": description, "last_update_est_str": last_update_est_str}
        return self.fear_and_greed

    def get_crypto_fear_and_greed(self) -> dict:
        """Get crypto fear and greed index data from alternative.me API and return as dictionary."""
        try:
            url = "https://api.alternative.me/fng/"
            params = {"limit": 1}
            response = requests.get(url, params=params)
            
            if response.ok:
                fng_api_data = response.json()
                data_entry = fng_api_data["data"][0]
                value = int(data_entry["value"])
                classification = data_entry["value_classification"]
                timestamp_utc = int(data_entry["timestamp"])
                
                # Convert timestamp to datetime in UTC, then to US/Eastern
                dt_utc = datetime.datetime.fromtimestamp(timestamp_utc, tz=datetime.timezone.utc)
                dt_est = dt_utc.astimezone(pytz.timezone("US/Eastern"))
                last_update_est_str = dt_est.strftime("%H:%M:%S  %Y-%m-%d %Z")
                
                self.crypto_fear_and_greed = {
                    "value": value, 
                    "description": classification, 
                    "last_update_est_str": last_update_est_str
                }
                return self.crypto_fear_and_greed
            else:
                raise Exception(f"API request failed: {response.status_code}")
                
        except Exception as e:
            self.crypto_fear_and_greed = {
                "value": None, 
                "description": "Error", 
                "last_update_est_str": "N/A",
                "error": str(e)
            }
            return self.crypto_fear_and_greed

    def get_fear_and_greed_html(self) -> str:
        """Get fear and greed index HTML for display."""
        try:
            fng_data = self.get_fear_and_greed()

            # Extract data
            value = fng_data["value"]
            description = fng_data["description"]
            last_update = fng_data["last_update_est_str"]

            # Convert value to 0-1 scale for gradient calculation
            normalized_value = value / 100.0
            
            # Calculate gradient color from red (0) to green (1)
            # Red component decreases as value increases
            red = int(255 * (1 - normalized_value))
            # Green component increases as value increases  
            green = int(255 * normalized_value)
            # Blue component stays low for better contrast
            blue = 50
            
            color_hex = f"#{red:02x}{green:02x}{blue:02x}"

            # Return HTML string
            html_content = f"""
            <div style="text-align:center;padding:10px;border-radius:10px;background-color:#f0f2f6;"> 
                <span style="margin:0;color:#262730;"><span style="color:{color_hex};font-size:20px;">●</span> {value} {description} (S&P 500)     <a href="https://www.cnn.com/markets/fear-and-greed" target="_blank" style="text-decoration:none;color:#2895f7;">
                        🔗 
                    </a></span><br> <span style="margin:0;color:#262730;">Last updated: {last_update}</span> 
                
            </div>
            """
            return html_content

        except Exception as e:
            return f'<div style="color:red;">Error loading Fear & Greed Index: {e}</div>'

    def get_crypto_fear_and_greed_html(self) -> str:
        """Get crypto fear and greed index HTML for display."""
        try:
            crypto_data = self.get_crypto_fear_and_greed()
            
            if "error" in crypto_data:
                return f'<div style="color:red;">Error loading Crypto Fear & Greed Index: {crypto_data["error"]}</div>'

            # Extract data
            value = crypto_data["value"]
            description = crypto_data["description"]
            last_update = crypto_data["last_update_est_str"]

            # Convert value to 0-1 scale for gradient calculation
            normalized_value = value / 100.0
            
            # Calculate gradient color from red (0) to green (1)
            # Red component decreases as value increases
            red = int(255 * (1 - normalized_value))
            # Green component increases as value increases  
            green = int(255 * normalized_value)
            # Blue component stays low for better contrast
            blue = 50
            
            color_hex = f"#{red:02x}{green:02x}{blue:02x}"

            # Return HTML string
            html_content = f"""
            <div style="text-align:center;padding:10px;border-radius:10px;background-color:#f0f2f6;"> 
                <span style="margin:0;color:#262730;"><span style="color:{color_hex};font-size:20px;">●</span> {value} {description} (Crypto)     <a href="https://alternative.me/crypto/fear-and-greed-index/" target="_blank" style="text-decoration:none;color:#2895f7;">
                        🔗 
                    </a></span><br> <span style="margin:0;color:#262730;">Last updated: {last_update}</span> 
                
            </div>
            """
            return html_content

        except Exception as e:
            return f'<div style="color:red;">Error loading Crypto Fear & Greed Index: {e}</div>'
