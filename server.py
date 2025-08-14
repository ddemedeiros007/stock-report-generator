# A simple Flask server to act as a proxy for the Alpha Vantage and Gemini APIs.
# To run this, you need to install Flask, flask_cors, requests, google-generativeai, and python-dotenv.
# You can do this with the command: pip install Flask flask_cors requests google-generativeai python-dotenv

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import requests
import re
import os
import time
import google.generativeai as genai

# --- NEW: Load environment variables from .env file ---
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, template_folder='templates')
CORS(app)  # Enable CORS for all routes

# IMPORTANT: Retrieve API keys from environment variables for security.
# Ensure these are set in your environment or a .env file (recommended for local development).
# Get a free Alpha Vantage key from https://www.alphavantage.co/
# Get a free Gemini API key from https://aistudio.google.com/app/apikey
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# --- DIAGNOSTIC PRINT STATEMENTS ---
print(
    f"DEBUG: Alpha Vantage API Key loaded: {'(not set)' if not ALPHA_VANTAGE_API_KEY else ALPHA_VANTAGE_API_KEY[:4] + '...' + ALPHA_VANTAGE_API_KEY[-4:]}")
print(
    f"DEBUG: Gemini API Key loaded: {'(not set)' if not GEMINI_API_KEY else GEMINI_API_KEY[:4] + '...' + GEMINI_API_KEY[-4:]}")

# Configure the Gemini API key
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY is not set. AI generation will not work.")

# Base URL for the Alpha Vantage API
AV_API_BASE_URL = 'https://www.alphavantage.co/query'


# --- Helper Functions for API Calls with Rate Limit Handling ---
def fetch_alpha_vantage_data(params):
    """
    Makes a request to Alpha Vantage API with exponential backoff for rate limiting.
    """
    if not ALPHA_VANTAGE_API_KEY:
        print("ERROR: Alpha Vantage API key is not configured.")
        return {"error": "Alpha Vantage API key is not configured in the server."}

    max_retries = 5
    initial_delay = 1  # seconds
    for i in range(max_retries):
        try:
            params['apikey'] = ALPHA_VANTAGE_API_KEY
            print(f"DEBUG: Making Alpha Vantage request to {AV_API_BASE_URL} with params: {params}")  # DIAGNOSTIC
            response = requests.get(AV_API_BASE_URL, params=params, timeout=10)  # 10 second timeout
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            print(f"DEBUG: Raw Alpha Vantage response data: {data}")  # DIAGNOSTIC

            # --- MODIFIED: Alpha Vantage specific error handling for rate limits or invalid calls ---
            if "Error Message" in data:
                if "rate limit" in data["Error Message"].lower():
                    print(f"Alpha Vantage Rate limit hit. Retrying in {initial_delay * (2 ** i)} seconds...")
                    time.sleep(initial_delay * (2 ** i))
                    continue  # Retry
                else:
                    return {"error": data["Error Message"]}
            elif "Information" in data and "rate limit" in data["Information"].lower():  # ADDED THIS CHECK
                print(f"Alpha Vantage 'Information' rate limit detected: {data['Information']}")
                return {"error": data["Information"]}  # Return the information as an error
            elif "Note" in data and "Thank you for using Alpha Vantage" in data["Note"]:
                pass  # This is often a warning, not an error if other data is present
            return data
        except requests.exceptions.RequestException as e:
            print(f"Network error during Alpha Vantage call: {e}. Retrying in {initial_delay * (2 ** i)} seconds...")
            time.sleep(initial_delay * (2 ** i))
        except Exception as e:
            print(f"Unexpected error during Alpha Vantage API call: {e}")
            return {"error": f"An unexpected error occurred during Alpha Vantage API call: {e}"}
    return {"error": "Max retries exceeded for Alpha Vantage API call due to rate limits or network issues."}


def generate_gemini_content(prompt_text):
    """
    Generates content using the Gemini API with exponential backoff.
    """
    if not GEMINI_API_KEY:
        print("ERROR: Gemini API key is not configured.")
        return {"error": "Gemini API key is not configured in the server."}

    max_retries = 5
    initial_delay = 1  # seconds
    for i in range(max_retries):
        try:
            model = genai.GenerativeModel(
                model_name='gemini-2.5-flash-preview-05-20',
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": {
                        "type": "OBJECT",
                        "properties": {
                            "fundamentals": {"type": "STRING"},
                            "management": {"type": "STRING"},
                            "guidance": {"type": "STRING"},
                            "competition": {"type": "STRING"},
                            "risks": {"type": "STRING"},
                            "targetPrice": {"type": "STRING"},
                            "revenueForecast": {"type": "STRING"},
                            "bottomLine": {"type": "STRING"},
                            "recommendation": {"type": "STRING"} # ADDED THIS LINE
                        }
                    }
                }
            )
            chat = model.start_chat(history=[])
            print(f"DEBUG: Sending prompt to Gemini: {prompt_text[:100]}...")  # DIAGNOSTIC
            response = chat.send_message(prompt_text)
            print(f"DEBUG: Raw Gemini response: {response.text[:100]}...")  # DIAGNOSTIC

            # Check for candidates and content before trying to access parts
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                return {"success": response.text}
            else:
                return {"error": "Gemini API response was empty or malformed."}

        except Exception as e:
            if "quota" in str(e).lower() or "rate limit" in str(e).lower() and retries > 0:
                print(f"Gemini Rate limit or quota hit. Retrying in {initial_delay * (2 ** i)} seconds...")
                time.sleep(initial_delay * (2 ** i))
                continue
            else:
                print(f"Error during Gemini API call: {e}")
                return {"error": f"An unexpected error occurred during AI content generation: {e}"}
    return {"error": "Max retries exceeded for Gemini API call due to rate limits or network issues."}


# The main route to serve the HTML file. Flask looks for files
# in a directory named 'templates'.
@app.route('/')
def index():
    return render_template('index.html')


# --- New Alpha Vantage Symbol Search Proxy Route ---
@app.route('/api/alpha-vantage/search/<keyword>', methods=['GET'])
def search_stock_symbol(keyword):
    sanitized_keyword = re.sub(r'[^a-zA-Z0-9.\-_]', '', keyword)
    if not sanitized_keyword:
        return jsonify({'error': 'Invalid search keyword provided.'}), 400

    search_params = {'function': 'SYMBOL_SEARCH', 'keywords': sanitized_keyword}
    search_data = fetch_alpha_vantage_data(search_params)

    if 'error' in search_data:
        print(f"DEBUG: Error from fetch_alpha_vantage_data: {search_data['error']}")  # DIAGNOSTIC
        return jsonify({'error': search_data['error']}), 500

    # Process bestMatches to return a cleaner list of options
    best_matches = search_data.get('bestMatches', [])
    print(f"DEBUG: Alpha Vantage bestMatches received: {best_matches}")  # DIAGNOSTIC

    # Filter out matches where the ticker symbol is empty or 'N/A'
    # And convert to a more user-friendly format
    options = []
    for match in best_matches:
        symbol = match.get('1. symbol')
        name = match.get('2. name')
        region = match.get('4. region')
        currency = match.get('8. currency')

        if symbol and symbol != 'N/A':  # Ensure symbol is valid
            options.append({
                'symbol': symbol,
                'name': name,
                'type': match.get('3. type'),
                'region': region,
                'currency': currency
            })

    # If no valid matches, return a specific error
    if not options:
        print("DEBUG: No valid options found after filtering Alpha Vantage matches.")  # DIAGNOSTIC
        # Only return 'No matching symbols found' if it's NOT due to an explicit API error like rate limit
        if not search_data.get('error'):  # Check if an error was explicitly set during the fetch
            return jsonify({'error': 'No matching symbols found for your search.'}), 404
        else:
            return jsonify({'error': search_data['error']}), 500  # Pass through the specific API error

    return jsonify(options)


# --- NEW: Individual Alpha Vantage Proxy Routes (to match frontend) ---
@app.route('/api/alpha-vantage/overview/<ticker>', methods=['GET'])
def get_overview(ticker):
    params = {'function': 'OVERVIEW', 'symbol': ticker}
    data = fetch_alpha_vantage_data(params)
    if 'error' in data:
        return jsonify({'error': data['error']}), 500
    return jsonify(data)

@app.route('/api/alpha-vantage/global-quote/<ticker>', methods=['GET'])
def get_global_quote(ticker):
    params = {'function': 'GLOBAL_QUOTE', 'symbol': ticker}
    data = fetch_alpha_vantage_data(params)
    if 'error' in data:
        return jsonify({'error': data['error']}), 500
    return jsonify(data)

@app.route('/api/alpha-vantage/income-statement/<ticker>', methods=['GET'])
def get_income_statement(ticker):
    params = {'function': 'INCOME_STATEMENT', 'symbol': ticker}
    data = fetch_alpha_vantage_data(params)
    if 'error' in data:
        return jsonify({'error': data['error']}), 500
    return jsonify(data)

@app.route('/api/alpha-vantage/earnings/<ticker>', methods=['GET'])
def get_earnings(ticker):
    params = {'function': 'EARNINGS', 'symbol': ticker}
    data = fetch_alpha_vantage_data(params)
    if 'error' in data:
        return jsonify({'error': data['error']}), 500
    return jsonify(data)

@app.route('/api/alpha-vantage/balance-sheet/<ticker>', methods=['GET'])
def get_balance_sheet(ticker):
    params = {'function': 'BALANCE_SHEET', 'symbol': ticker}
    data = fetch_alpha_vantage_data(params)
    if 'error' in data:
        return jsonify({'error': data['error']}), 500
    return jsonify(data)

@app.route('/api/alpha-vantage/cash-flow/<ticker>', methods=['GET'])
def get_cash_flow(ticker):
    params = {'function': 'CASH_FLOW', 'symbol': ticker}
    data = fetch_alpha_vantage_data(params)
    if 'error' in data:
        return jsonify({'error': data['error']}), 500
    return jsonify(data)

# --- Consolidated full-report route (kept for reference, but frontend now uses individual) ---
@app.route('/api/alpha-vantage/full-report/<ticker>', methods=['GET'])
def get_full_stock_report(ticker):
    sanitized_ticker = re.sub(r'[^a-zA-Z0-9.\-_]', '', ticker).upper()
    if not sanitized_ticker:
        return jsonify({'error': 'Invalid ticker symbol provided.'}), 400

    report_data = {
        'symbol': sanitized_ticker,
        'global_quote': {},
        'overview': {},
        'income_statement': [],
        'earnings': {},
        'error': None
    }

    # 1. Fetch Global Quote
    quote_params = {'function': 'GLOBAL_QUOTE', 'symbol': sanitized_ticker}
    quote_data = fetch_alpha_vantage_data(quote_params)
    if 'error' in quote_data:
        report_data['error'] = report_data.get('error', '') + f"Quote Error: {quote_data['error']}. "
    else:
        report_data['global_quote'] = quote_data.get('Global Quote', {})

    # 2. Fetch Company Overview
    overview_params = {'function': 'OVERVIEW', 'symbol': sanitized_ticker}
    overview_data = fetch_alpha_vantage_data(overview_params)
    if 'error' in overview_data:
        report_data['error'] = report_data.get('error', '') + f"Overview Error: {overview_data['error']}. "
    else:
        report_data['overview'] = overview_data

    # 3. Fetch Income Statement (last few annual reports)
    income_params = {'function': 'INCOME_STATEMENT', 'symbol': sanitized_ticker}
    income_data = fetch_alpha_vantage_data(income_params)
    if 'error' in income_data:
        report_data['error'] = report_data.get('error', '') + f"Income Statement Error: {income_data['error']}. "
    else:
        report_data['income_statement'] = income_data.get('annualReports', [])[:3]

    # 4. Fetch Earnings (Annual)
    earnings_params = {'function': 'EARNINGS', 'symbol': sanitized_ticker}
    earnings_data = fetch_alpha_vantage_data(earnings_params)
    if 'error' in earnings_data:
        report_data['error'] = report_data.get('error', '') + f"Earnings Error: {earnings_data['error']}. "
    else:
        report_data['earnings']['annualReports'] = earnings_data.get('annualEarnings', [])[:3]

    # Check for primary data presence
    if not report_data['global_quote'] and not report_data['overview']:
        final_error = report_data.get('error', '')
        if not final_error:
            final_error = "Stock ticker not found or no data available from Alpha Vantage."
        # If there's an existing error from any fetch, use that, otherwise use the generic not found message
        return jsonify({'error': report_data['error'] or final_error}), 404  # Ensure existing error is propagated

    return jsonify(report_data)


# --- New Gemini Proxy Route ---
@app.route('/api/gemini/generate-report-sections', methods=['POST'])
def generate_report_sections():
    data = request.json
    # The prompt is now structured differently in the frontend payload
    # It's inside contents[0].parts[0].text
    prompt_from_frontend = data.get('contents')[0]['parts'][0]['text'] if data.get('contents') and data['contents'][
        0].get('parts') else None

    if not prompt_from_frontend:
        return jsonify({'error': 'No prompt provided for AI generation.'}), 400

    ai_response = generate_gemini_content(prompt_from_frontend)

    if 'error' in ai_response:
        return jsonify({'error': ai_response['error']}), 500

    # The response.text from Gemini will already be a JSON string due to response_mime_type
    return jsonify({'generated_content': ai_response['success']})


if __name__ == '__main__':
    # Set host to '0.0.0.0' to make the server accessible externally
    app.run(debug=True, host='0.0.0.0', port=5000)
