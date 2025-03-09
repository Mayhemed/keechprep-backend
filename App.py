from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import os
import tempfile
import re
import json
import math
from datetime import datetime
import logging
import sys
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()
from services.ai_service import (
    generate_keech_declaration, parse_custom_instructions, should_apply_instruction, 
    ai_redact_timesheet, analyze_discount_entries, llm_client,
    review_keech_entries, improve_keech_entries  # Add these new imports
)
from services.threaded_ai_service import threaded_redact_timesheet, threaded_batch_review_entries
from services.evidence_processor import get_evidence_for_period
import concurrent.futures

from utils.loggers import setup_logger

# Set up logging
logger = setup_logger()
logger.info("Starting the application...")


# Add this near the start of your app, after imports
# Create necessary directories
os.makedirs('case_evidence', exist_ok=True)
logger.info("Created case_evidence directory")



# Timesheet processing functions
def process_timesheet(file_path, period_type='weekly'):
    """
    Process the timesheet data and organize by week or month
    """
    logger.info(f"Processing timesheet from {file_path} with period_type={period_type}")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    logger.info(f"CSV loaded, found columns: {df.columns.tolist()}")
    
    # First, get the current column order to preserve after renaming
    original_column_order = df.columns.tolist()
    logger.info(f"Original column order: {original_column_order}")
    
    # Map our specific fields to standard names
    field_mapping = {
        'Date': 'Date',               # Keep as is
        'Activity Description': 'Description',
        'Quantity': 'Hours',
        'Rate': 'Rate',               # Keep as is
        'Activity User': 'Timekeeper'
    }
    
    # Apply mapping for columns that exist
    rename_dict = {}
    for orig_col, new_col in field_mapping.items():
        if orig_col in df.columns:
            rename_dict[orig_col] = new_col
    
    # Rename columns according to our mapping
    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)
        logger.info(f"Renamed columns: {rename_dict}")
    
    # Create a new column order with the renamed columns but in the original order
    new_column_order = []
    renamed_map = {orig: new for orig, new in rename_dict.items()}
    
    for col in original_column_order:
        if col in renamed_map:
            new_column_order.append(renamed_map[col])
        else:
            new_column_order.append(col)
    
    # Reorder columns to match the original order
    df = df[new_column_order]
    logger.info(f"Reordered columns after renaming: {df.columns.tolist()}")
    
    # Ensure date column is datetime with custom date parsing for complex formats
    if 'Date' in df.columns:
        try:
            # Try standard conversion first
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Check for any NaT (not a time) values which indicate parsing failed
            nat_count = df['Date'].isna().sum()
            if nat_count > 0:
                logger.warning(f"Found {nat_count} invalid date formats. Implementing custom date parsing.")
                
                # Custom parsing for complex date formats
                def parse_complex_date(date_str):
                    if pd.isna(date_str):
                        return pd.NaT
                    
                    try:
                        # Try standard parsing first
                        return pd.to_datetime(date_str)
                    except:
                        # For formats like "04/24-26/2024", take the first date
                        import re
                        if isinstance(date_str, str) and '-' in date_str:
                            # Extract the month and year
                            match = re.match(r'(\d{1,2})/(\d{1,2})-\d{1,2}/(\d{4})', date_str)
                            if match:
                                month, day, year = match.groups()
                                return pd.to_datetime(f"{month}/{day}/{year}")
                            
                            # Try another pattern like MM/DD-DD/YYYY
                            match = re.match(r'(\d{1,2})/(\d{1,2})-(\d{1,2})/(\d{4})', date_str)
                            if match:
                                month, day1, _, year = match.groups()
                                return pd.to_datetime(f"{month}/{day1}/{year}")
                        
                        # Default fallback - return the first day of the current month
                        logger.warning(f"Could not parse date: {date_str}. Using fallback.")
                        return pd.to_datetime('today').replace(day=1)
                
                # Apply custom parsing to problematic dates
                orig_dates = df['Date'].copy()
                df['Date'] = df['Date'].fillna(orig_dates)  # Restore original values
                df['Date'] = df['Date'].apply(parse_complex_date)
            
            logger.info("Converted Date to datetime format")
        except Exception as e:
            logger.error(f"Error parsing dates: {str(e)}")
            # If conversion fails, create a dummy date column
            df['Date'] = pd.to_datetime('today')
            logger.warning("Using today's date as fallback for all entries due to parsing errors")
    else:
        # Try to find a date column
        date_found = False
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if not df[col].isna().all():  # If at least some values converted successfully
                        df.rename(columns={col: 'Date'}, inplace=True)
                        logger.info(f"Found and converted date column: {col}")
                        date_found = True
                        break
                except:
                    continue
        
        # If no date column found, create a dummy one
        if not date_found:
            logger.warning("No date column found. Creating a dummy date column.")
            df['Date'] = pd.to_datetime('today')
            logger.warning("Using today's date for all entries")
    
    # Log current columns
    logger.info(f"Columns after mapping: {df.columns.tolist()}")
    
    # Ensure necessary columns exist, handling edge cases if columns are missing
    required_columns = ['Date', 'Hours', 'Description']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.warning(f"Missing required columns: {missing_columns}")
    
    # If Hours not found but Quantity exists, use it
    if 'Hours' not in df.columns and 'Quantity' in df.columns:
        df.rename(columns={'Quantity': 'Hours'}, inplace=True)
        logger.info("Using Quantity as Hours")
    
    # If Description not found but Activity Description exists, use it
    if 'Description' not in df.columns and 'Activity Description' in df.columns:
        df.rename(columns={'Activity Description': 'Description'}, inplace=True)
        logger.info("Using Activity Description as Description")
    
    # If we have Hours and Rate but no Amount, calculate it
    if 'Hours' in df.columns and 'Rate' in df.columns and 'Amount' not in df.columns:
        df['Amount'] = df['Hours'] * df['Rate']
        logger.info("Calculated Amount from Hours and Rate")
    
    # Group by period
    if period_type == 'weekly':
        df['Period'] = df['Date'].dt.to_period('W')
        logger.info("Grouped data weekly")
    else:  # monthly
        df['Period'] = df['Date'].dt.to_period('M')
        logger.info("Grouped data monthly")
    
    # Convert Period to string for JSON serialization
    df['Period'] = df['Period'].astype(str)
    
    # Group and aggregate data
    agg_dict = {'Hours': 'sum'}
    if 'Amount' in df.columns:
        agg_dict['Amount'] = 'sum'
    
    # This is a DataFrame containing aggregated data per period
    period_df = df.groupby('Period').agg(agg_dict).reset_index()
    
    detailed_data = []
    for period in df['Period'].unique():
        period_entries = df[df['Period'] == period].sort_values('Date')
        
        # Handle entries safely
        entries = []
        for _, row in period_entries.iterrows():
            entry = {}
            for col in row.index:
                entry[col] = None if pd.isna(row[col]) else row[col]
            entries.append(entry)
        
        # Prepare the period summary with safe null handling
        # This is a dictionary for a single period
        period_summary = {
            'period': period,
            'entries': entries
        }
        
        # Handle Hours with proper error handling
        try:
            if 'Hours' in period_entries.columns:
                total_hours = period_entries['Hours'].sum(skipna=True)
                if pd.isna(total_hours):
                    period_summary['total_hours'] = 0.0
                    logger.warning(f"Period {period} has NaN for total hours, using 0")
                else:
                    period_summary['total_hours'] = float(total_hours)
            else:
                period_summary['total_hours'] = 0.0
                logger.warning(f"No Hours column found for period {period}, using 0")
        except Exception as e:
            logger.error(f"Error calculating total hours for period {period}: {str(e)}")
            period_summary['total_hours'] = 0.0
        
        # Add amount if it exists with proper error handling
        try:
            if 'Amount' in df.columns and 'Amount' in period_entries.columns:
                total_amount = period_entries['Amount'].sum(skipna=True)
                if pd.isna(total_amount):
                    period_summary['total_amount'] = None
                    logger.warning(f"Period {period} has NaN for total amount, using None")
                else:
                    period_summary['total_amount'] = float(total_amount)
            else:
                period_summary['total_amount'] = None
        except Exception as e:
            logger.error(f"Error calculating total amount for period {period}: {str(e)}")
            period_summary['total_amount'] = None
            
        detailed_data.append(period_summary)
    
    # Convert period_df safely - this is the DataFrame with period aggregates
    try:
        periods = []
        # Check if period_df is a DataFrame
        if isinstance(period_df, pd.DataFrame):
            for _, row in period_df.iterrows():
                period_dict = {}
                for col in row.index:
                    period_dict[col] = None if pd.isna(row[col]) else row[col]
                periods.append(period_dict)
        else:
            # If it's not a DataFrame (possibly a dict), handle accordingly
            logger.warning("period_df is not a DataFrame, using empty list for periods")
            periods = []
    except Exception as e:
        logger.error(f"Error processing period_df: {str(e)}")
        periods = []
    
    # Convert raw_data safely
    raw_data = []
    for _, row in df.iterrows():
        row_dict = {}
        for col in row.index:
            row_dict[col] = None if pd.isna(row[col]) else row[col]
        raw_data.append(row_dict)
    
    # Calculate total hours with error handling
    try:
        if 'Hours' in df.columns:
            total_hours = df['Hours'].sum(skipna=True)
            if pd.isna(total_hours):
                total_hours = 0.0
                logger.warning("Total hours is NaN, using 0")
            else:
                total_hours = float(total_hours)
        else:
            total_hours = 0.0
            logger.warning("No Hours column found, using 0 for total")
    except Exception as e:
        logger.error(f"Error calculating total hours: {str(e)}")
        total_hours = 0.0
    
    # Calculate total amount with error handling
    try:
        if 'Amount' in df.columns:
            total_amount = df['Amount'].sum(skipna=True)
            if pd.isna(total_amount):
                total_amount = None
                logger.warning("Total amount is NaN, using None")
            else:
                total_amount = float(total_amount)
        else:
            total_amount = None
    except Exception as e:
        logger.error(f"Error calculating total amount: {str(e)}")
        total_amount = None
    
    result = {
        'periods': periods,
        'detailed_data': detailed_data,
        'total_hours': total_hours,
        'total_amount': total_amount,
        'raw_data': raw_data
    }
    
    logger.info(f"Processing complete. Total hours: {result['total_hours']}")
    return result

def redact_timesheet(processed_data, custom_instructions=None):
    """
    Create a redacted version of the timesheet with sensitive info removed
    Uses AI-powered redaction when available
    
    Args:
        processed_data: The processed timesheet data
        custom_instructions: Any special instructions for redaction
        
    Returns:
        A redacted copy of the timesheet data
    """
    logger.info("Redacting timesheet data")
    
    # Use the AI-powered redaction service
    try:
        redacted_data = ai_redact_timesheet(processed_data, custom_instructions)
        logger.info("AI-powered redaction completed successfully")
        return redacted_data
    except Exception as e:
        # If AI redaction fails, log the error and return the original data
        logger.error(f"AI redaction failed: {str(e)}")
        logger.info("AI redaction failed - returning original data")
        
        # Create a deep copy to avoid modifying the original
        import copy
        return copy.deepcopy(processed_data)
        
        # Not using pattern-based redaction as requested
        """
        # Patterns that might indicate privileged information
        patterns = [
            r'confidential',
            r'privileged',
            r'attorney.client',
            r'strategy',
            r'advice',
            r'counsel.*regarding',
            r'internal.*discussion'
        ]
        
        combined_pattern = re.compile('|'.join(patterns), re.IGNORECASE)
        
        # Redact sensitive information in detailed data
        for period_data in redacted_data['detailed_data']:
            for entry in period_data['entries']:
                description = entry.get('Description', '')
                if description and isinstance(description, str):
                    # Check if the description contains sensitive information
                    if combined_pattern.search(description):
                        # Redact the sensitive parts
                        redacted_desc = combined_pattern.sub('[REDACTED]', description)
                        entry['Description'] = redacted_desc
                        entry['RedactionReason'] = 'Pattern-based redaction'
        
        # Also redact in raw data
        for entry in redacted_data['raw_data']:
            description = entry.get('Description', '')
            if description and isinstance(description, str):
                if combined_pattern.search(description):
                    entry['Description'] = combined_pattern.sub('[REDACTED]', description)
        
        logger.info("Pattern-based redaction complete")
        return redacted_data
        """

# This function has been moved to services/ai_service.py

# These functions have been moved to services/ai_service.py

# Flask application
app = Flask(__name__)
CORS(app)  # Allow all origins

# Custom JSON encoder to handle NaN values
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            elif pd.isna(obj):
                return None
            elif isinstance(obj, pd.Series):
                return obj.apply(lambda x: None if pd.isna(x) else x).to_dict()
            elif isinstance(obj, pd.DataFrame):
                result = []
                for _, row in obj.iterrows():
                    row_dict = {}
                    for col in row.index:
                        row_dict[col] = None if pd.isna(row[col]) else row[col]
                    result.append(row_dict)
                return result
            elif isinstance(obj, pd.Period):
                return str(obj)
            elif hasattr(obj, 'isoformat'):  # Handle datetime objects
                return obj.isoformat()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return super().default(obj)
        except Exception as e:
            # If we can't handle it, convert to string as a fallback
            logger.warning(f"JSON encoder error, converting to string: {e}")
            return str(obj)

app.json_encoder = CustomJSONEncoder
logger.info("Flask app initialized with CORS support")

@app.route('/api/upload', methods=['POST'])
def upload_timesheet():
    """
    Endpoint to upload and process a timesheet CSV file
    """
    logger.info("Received upload request")
    try:
        logger.info(f"Request files: {request.files}")
        logger.info(f"Request form: {request.form}")
        logger.info(f"Request content type: {request.content_type}")
        
        if 'file' not in request.files:
            logger.warning("No file part in request")
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        logger.info(f"Received file: {file.filename}, {file.content_type}")
        
        if file.filename == '':
            logger.warning("No file selected")
            return jsonify({"error": "No file selected"}), 400
        
        # Create temporary file to store the uploaded CSV
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        file.save(temp_file.name)
        logger.info(f"File saved to temporary location: {temp_file.name}")
        
        # Store file path in session for later processing
        file_path = temp_file.name
        
        # Preview the data (first 5 rows)
        try:
            # Desired columns in specific order - MUST preserve this order
            desired_columns = ['Matter', 'Date', 'Activity Code', 'Activity Description', 
                              'Rate', 'Quantity', 'Type', 'Activity User', 'Non-Billable']
            
            # Read all columns first
            df = pd.read_csv(file_path)
            logger.info(f"CSV loaded successfully with {len(df)} rows and {len(df.columns)} columns")
            
            # Get the actual columns in the file
            actual_columns = df.columns.tolist()
            logger.info(f"Actual columns in file: {actual_columns}")
            
            # Filter to only include desired columns that exist in the file
            # CRITICALLY IMPORTANT: preserve the order from desired_columns
            filtered_columns = [col for col in desired_columns if col in actual_columns]
            
            # If we have no matching columns, use all columns
            if not filtered_columns:
                logger.warning("None of the desired columns found, using all columns")
                filtered_columns = actual_columns
            else:
                logger.info(f"Using filtered columns: {filtered_columns}")
            
            # Select only the desired columns - IN THE EXACT ORDER SPECIFIED
            filtered_df = df[filtered_columns]
            
            # To ensure the order is maintained:
            logger.info(f"Column order after filtering: {filtered_df.columns.tolist()}")
            
            # Get a preview without modifying the data
            preview_df = filtered_df.head(5)
            
            # Convert to dictionary and handle NaN values during conversion
            # Use OrderedDict to guarantee column order is preserved
            from collections import OrderedDict
            
            preview = []
            for _, row in preview_df.iterrows():
                # Use OrderedDict to guarantee order
                row_dict = OrderedDict()
                for col in filtered_columns:  # Iterate in the exact order we want
                    if pd.isna(row[col]):
                        row_dict[col] = None
                    else:
                        row_dict[col] = row[col]
                preview.append(row_dict)
                
            # Log the keys of the first preview item to verify order
            if preview:
                logger.info(f"Preview column order: {list(preview[0].keys())}")
            
            logger.info(f"File preview generated, filtered columns: {filtered_columns}")
            logger.info(f"First 5 rows preview created")
            
            # Process the data to get available periods
            try:
                # First process the data with weekly periods
                processed_data_weekly = process_timesheet(file_path, 'weekly')
                weekly_periods = [period_data.get('period') for period_data in processed_data_weekly.get('detailed_data', [])]
                
                # Then process the data with monthly periods
                processed_data_monthly = process_timesheet(file_path, 'monthly')
                monthly_periods = [period_data.get('period') for period_data in processed_data_monthly.get('detailed_data', [])]
                
                # Log the periods
                logger.info(f"Available weekly periods: {weekly_periods}")
                logger.info(f"Available monthly periods: {monthly_periods}")
            except Exception as period_error:
                logger.error(f"Error getting periods: {str(period_error)}")
                weekly_periods = []
                monthly_periods = []
            
            # Replace the original dataframe with the filtered one for further processing
            df = filtered_df
        except Exception as csv_error:
            logger.error(f"Error processing CSV: {str(csv_error)}")
            return jsonify({"error": f"Error processing CSV: {str(csv_error)}"}), 400
        
        # Get available AI models
        from services.ai_service import llm_client
        ai_models = llm_client.get_flat_model_list()
        
        response_data = {
            "message": "File uploaded successfully",
            "preview": preview,
            "columns": filtered_columns,  # Use the filtered columns list
            "original_columns": actual_columns,  # Include the original columns for reference
            "file_path": file_path,
            "total_rows": len(df),
            "filename": file.filename,
            "available_periods": {
                "weekly": weekly_periods,
                "monthly": monthly_periods
            },
            "ai_info": {
                "models": ai_models,
                "openai_key_available": bool(llm_client.openai_api_key),
                "anthropic_key_available": bool(llm_client.anthropic_api_key)
            }
        }
        
        logger.info(f"Returning successful response")
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_timesheet():
    """
    Endpoint to analyze the uploaded timesheet with threading for better performance
    """
    logger.info("Received analyze request")
    try:
        data = request.json
        file_path = data.get('file_path')
        period_type = data.get('period_type', 'weekly')  # weekly or monthly
        custom_instructions = data.get('custom_instructions', '')
        ai_provider = data.get('ai_provider')  # Optional AI provider parameter
        ai_model = data.get('ai_model')  # Optional AI model parameter
        selected_period = data.get('selected_period')  # Optional specific period to process
        use_threading = data.get('use_threading', True)  # New parameter to enable/disable threading
        max_workers = data.get('max_workers', 5)  # Number of concurrent threads
        
        # Set environment variables for the AI service if provided
        if ai_provider:
            os.environ['AI_PROVIDER'] = ai_provider
            logger.info(f"Setting AI provider to: {ai_provider}")
        if ai_model:
            os.environ['AI_MODEL'] = ai_model
            logger.info(f"Setting AI model to: {ai_model}")
        
        logger.info(f"Analyzing timesheet: {file_path}, period_type: {period_type}, threading: {use_threading}")
        
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return jsonify({"error": "File not found"}), 404
        
        # Process the timesheet
        processed_data = process_timesheet(file_path, period_type)
        
        # If a specific period was selected, filter the processed data to only include that period
        if selected_period:
            logger.info(f"Filtering data to only include period: {selected_period}")
            
            # Filter detailed_data
            filtered_detailed_data = [
                period_data for period_data in processed_data['detailed_data'] 
                if period_data.get('period') == selected_period
            ]
            
            if not filtered_detailed_data:
                logger.warning(f"No data found for period: {selected_period}")
                return jsonify({"error": f"No data found for period: {selected_period}"}), 404
                
            # Create a filtered version of processed_data
            filtered_processed_data = {
                'detailed_data': filtered_detailed_data,
                'periods': [p for p in processed_data['periods'] if p.get('Period') == selected_period],
                'total_hours': sum(period_data.get('total_hours', 0) for period_data in filtered_detailed_data),
                'total_amount': sum(period_data.get('total_amount', 0) for period_data in filtered_detailed_data if period_data.get('total_amount')),
                'raw_data': [
                    entry for entry in processed_data.get('raw_data', [])
                    if entry.get('Period') == selected_period
                ]
            }
            
            # Use the filtered data for declaration and redaction
            processed_data_for_analysis = filtered_processed_data
            logger.info(f"Filtered data contains {len(filtered_detailed_data)} periods")
        else:
            # Use all data
            processed_data_for_analysis = processed_data
        
        # Generate Keech declaration
        keech_declaration = generate_keech_declaration(
            processed_data_for_analysis, 
            period_type,
            custom_instructions
        )
        
        # Get the task type - default to "redact" if not specified
        task = data.get('task', 'redact')  # Can be 'redact' or 'discount'
        logger.info(f"Task type: {task}")
        
        if task == 'redact':
            # Choose between threaded or regular redaction based on the use_threading parameter
            if use_threading:
                logger.info(f"Using threaded redaction with {max_workers} workers")
                result_data = threaded_redact_timesheet(
                    processed_data_for_analysis, 
                    custom_instructions,
                    max_workers=max_workers
                )
            else:
                # Use the original non-threaded function
                result_data = ai_redact_timesheet(processed_data_for_analysis, custom_instructions)
                
        elif task == 'discount':
            # For discount task, use the existing method for now
            # You could implement a threaded version similar to redaction
            result_data = analyze_discount_entries(processed_data_for_analysis, custom_instructions)
        else:
            # Default to redaction
            if use_threading:
                result_data = threaded_redact_timesheet(processed_data_for_analysis, custom_instructions)
            else:
                result_data = ai_redact_timesheet(processed_data_for_analysis, custom_instructions)
            task = 'redact'
        
        # Get AI model and provider that were used
        ai_model = os.getenv('AI_MODEL', 'None specified')
        ai_provider = os.getenv('AI_PROVIDER', 'openai')
        
        # Initialize variables for task-specific data
        ai_redaction_used = False
        redaction_used = False
        redaction_counts = {'ai': 0, 'pattern': 0, 'total': 0, 'debug': 0, 'error': 0}
        discount_info = None
        model_issue = None
        
        # Task-specific processing
        if task == 'redact':
            # Check if any redactions were applied
            if result_data and 'detailed_data' in result_data:
                for period_data in result_data['detailed_data']:
                    for entry in period_data.get('entries', []):
                        # Check if this entry has any redaction
                        if 'Description' in entry and entry['Description']:
                            # Look for typical redaction markers
                            if '[REDACTED]' in entry['Description']:
                                redaction_used = True
                                redaction_counts['total'] += 1
                                
                                # Determine if AI or pattern-based
                                if 'RedactionReason' in entry:
                                    if 'pattern-based' in entry['RedactionReason'].lower():
                                        redaction_counts['pattern'] += 1
                                    else:
                                        redaction_counts['ai'] += 1
                            
                            # Look for debug/error indicators
                            if '[O-SERIES MODEL LIMITATION]' in entry['Description']:
                                redaction_counts['debug'] += 1
                                model_issue = 'o-series-limitation'
                            elif '[DEBUG]' in entry['Description'] or '[FAILED]' in entry['Description'] or '[ERROR]' in entry['Description']:
                                redaction_counts['error'] += 1
                                model_issue = 'model-error'
            
            # Consider it AI redaction if at least some entries were redacted by AI
            ai_redaction_used = redaction_counts['ai'] > 0
            
            # Log redaction statistics
            logger.info(f"Redaction statistics: {redaction_counts}")
            logger.info(f"AI redaction used: {ai_redaction_used}")
            logger.info(f"Any redaction applied: {redaction_used}")
            
            # Log any model issues detected
            if model_issue:
                logger.warning(f"Model issue detected: {model_issue}")
            
            # If we detected O-series limitations, add a warning to the response
            if model_issue == 'o-series-limitation':
                logger.warning("O-series model limitation detected - adding warning to response")
                # Will be added to the response later
            
        elif task == 'discount':
            # Extract discount information if available
            if 'discount_summary' in result_data:
                discount_info = result_data['discount_summary']
                logger.info(f"Discount analysis complete. Entries discounted: {discount_info['entries_discounted']}")
                if discount_info['hours_saved'] > 0:
                    logger.info(f"Hours saved: {discount_info['hours_saved']} ({discount_info['percentage_saved']:.2f}%)")
        
        # Extract AI debug information
        ai_debug = None
        if 'ai_debug' in result_data:
            ai_debug = {
                'prompt': result_data['ai_debug']['prompts'][0] if result_data['ai_debug']['prompts'] else None,
                'response': result_data['ai_debug']['responses'][0] if result_data['ai_debug']['responses'] else None
            }
            # Remove the debug info from the result data to avoid sending too much data
            del result_data['ai_debug']
        
        # Build the response object
        response_data = {
            "keech_declaration": keech_declaration,
            "task": task,
            "ai_info": {
                "provider": ai_provider,
                "model": ai_model,
                "openai_key_available": bool(llm_client.openai_api_key),
                "anthropic_key_available": bool(llm_client.anthropic_api_key),
                "threading_used": use_threading,
                "max_workers": max_workers if use_threading else 0
            },
            "ai_debug": ai_debug,
            "summary": {
                "total_hours": processed_data_for_analysis.get('total_hours', 0),
                "total_amount": processed_data_for_analysis.get('total_amount'),
                "period_type": period_type
            }
        }
        
        # Add task-specific data to the response
        if task == 'redact':
            response_data["redacted_data"] = result_data
            response_data["ai_info"]["redaction_used"] = ai_redaction_used
            
            # Add redaction statistics
            response_data["redaction_stats"] = redaction_counts
            
            # Add model issue information if detected
            if model_issue:
                response_data["model_issue"] = {
                    "type": model_issue,
                    "message": "The selected model returned empty responses. O-series models like GPT-4o and GPT-o1 may not work well with redaction tasks. Please try a different model like GPT-4 or Claude."
                }
                
                # If we detected O-series limitations, add a specific warning
                if model_issue == 'o-series-limitation':
                    response_data["model_issue"]["message"] = "OpenAI O-series models (like o1) currently don't work well with redaction tasks and return empty responses. Please try a different model like GPT-4 or Claude."
                elif model_issue == 'model-error':
                    response_data["model_issue"]["message"] = "The selected model encountered errors during processing. Please try a different model."
                
        elif task == 'discount':
            response_data["discount_data"] = result_data
            if discount_info:
                response_data["discount_summary"] = discount_info
        
        logger.info(f"Analysis complete for task: {task}")
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error analyzing timesheet: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Handle timeout errors specially
        error_message = str(e)
        if "timed out" in error_message.lower():
            return jsonify({
                "error": "The AI service timed out. This usually means the model is too busy or the input is too complex. Please try a different AI model or try again later.",
                "error_details": error_message,
                "error_type": "timeout"
            }), 504  # Gateway Timeout status code
        
        # Handle other errors
        return jsonify({"error": error_message}), 500

@app.route('/api/upload-case-evidence', methods=['POST'])
def upload_case_evidence():
    """
    Endpoint to upload case evidence document
    
    This evidence will be used later for improving time entries
    """
    logger.info("Received case evidence upload request")
    try:
        logger.info(f"Request files: {request.files}")
        
        if 'file' not in request.files:
            logger.warning("No file part in request")
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        logger.info(f"Received file: {file.filename}, {file.content_type}")
        
        if file.filename == '':
            logger.warning("No file selected")
            return jsonify({"error": "No file selected"}), 400
        
        # Create a temporary directory for case evidence if not exists
        evidence_dir = os.path.join(os.getcwd(), 'case_evidence')
        os.makedirs(evidence_dir, exist_ok=True)
        
        # Save the file with a timestamp to avoid name conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = re.sub(r'[^\w\-_\.]', '_', file.filename)  # Sanitize filename
        evidence_filename = f"{timestamp}_{safe_filename}"
        evidence_path = os.path.join(evidence_dir, evidence_filename)
        
        file.save(evidence_path)
        logger.info(f"Case evidence saved to: {evidence_path}")
        
        # Read file content for preview
        file_content = ""
        try:
            with open(evidence_path, 'r', encoding='utf-8') as f:
                # Read first 1000 characters for preview
                file_content = f.read(1000)
                if len(file_content) >= 1000:
                    file_content += "..."
        except Exception as e:
            logger.warning(f"Could not read file content for preview: {str(e)}")
            file_content = "Content preview not available"
        
        return jsonify({
            "message": "Case evidence uploaded successfully",
            "file_path": evidence_path,
            "filename": file.filename,
            "content_preview": file_content
        })
    
    except Exception as e:
        logger.error(f"Error uploading case evidence: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/improve-keech-entries', methods=['POST'])
def improve_keech_entries_api():
    """
    Endpoint to improve time entries for a Keech Declaration
    based on review feedback and case evidence
    """
    logger.info("Received Keech improvement request")
    try:
        data = request.json
        file_path = data.get('file_path')
        period_type = data.get('period_type', 'weekly')
        selected_period = data.get('selected_period')
        review_feedback = data.get('review_feedback')
        evidence_path = data.get('evidence_path')
        ai_provider = data.get('ai_provider')
        ai_model = data.get('ai_model')
        use_threading = data.get('use_threading', True)
        max_workers = data.get('max_workers', 5)
        
        # Set environment variables for the AI service if provided
        if ai_provider:
            os.environ['AI_PROVIDER'] = ai_provider
            logger.info(f"Setting AI provider to: {ai_provider}")
        if ai_model:
            os.environ['AI_MODEL'] = ai_model
            logger.info(f"Setting AI model to: {ai_model}")
        
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return jsonify({"error": "File not found"}), 404
            
        if not review_feedback:
            logger.warning("No review feedback provided")
            return jsonify({"error": "Review feedback is required for improvement"}), 400
        
        # Process the timesheet to get the time entries
        processed_data = process_timesheet(file_path, period_type)
        
        # If a specific period was selected, filter the data
        if selected_period:
            logger.info(f"Filtering data to only include period: {selected_period}")
            
            # Filter detailed_data
            filtered_detailed_data = [
                period_data for period_data in processed_data['detailed_data'] 
                if period_data.get('period') == selected_period
            ]
            
            if not filtered_detailed_data:
                logger.warning(f"No data found for period: {selected_period}")
                return jsonify({"error": f"No data found for period: {selected_period}"}), 404
                
            # Extract entries from the filtered period
            time_entries = []
            for period_data in filtered_detailed_data:
                time_entries.extend(period_data.get('entries', []))
        else:
            # Get all entries from all periods
            time_entries = []
            for period_data in processed_data['detailed_data']:
                time_entries.extend(period_data.get('entries', []))
        
        if not time_entries:
            logger.warning("No time entries found for improvement")
            return jsonify({"error": "No time entries found for improvement"}), 400
        
        # Get period-specific evidence if available
        filtered_evidence = None
        if evidence_path and os.path.exists(evidence_path):
            try:
                # Filter evidence based on the selected period
                if selected_period:
                    filtered_evidence = get_evidence_for_period(evidence_path, selected_period)
                    logger.info(f"Filtered evidence for period {selected_period}")
                else:
                    # Use the full evidence file if no specific period is selected
                    with open(evidence_path, 'r', encoding='utf-8') as f:
                        filtered_evidence = f.read()
                    logger.info(f"Using full evidence (no specific period selected)")
            except Exception as e:
                logger.warning(f"Error reading case evidence: {str(e)}")
                filtered_evidence = None
        
        logger.info(f"Improving {len(time_entries)} time entries")
        
        # Call the improvement function with threaded option
        if use_threading and len(time_entries) > 10:
            from services.threaded_ai_service import ThreadedAIProcessor
            
            # Create processor instance
            processor = ThreadedAIProcessor(max_workers=max_workers)
            
            # Define function to process each entry
            def improve_entry(entry):
                from services.ai_service_entry import improve_single_entry
                return improve_single_entry(entry, review_feedback, filtered_evidence)
            
            # Process entries in parallel
            logger.info(f"Using threaded processing with {max_workers} workers")
            improved_entries = processor.process_entries_in_parallel(
                time_entries, 
                improve_entry
            )
            
            # Generate CSV content
            import pandas as pd
            from io import StringIO
            
            # Convert improved entries to DataFrame and then to CSV
            df = pd.DataFrame(improved_entries)
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            # Generate a summary review
            improvement_result = {
                "status": "success",
                "improved_entries": csv_content,
                "final_review": f"Successfully improved {len(improved_entries)} entries using parallel processing with {max_workers} workers.",
                "full_response": f"Threaded processing completed. Improved {len(improved_entries)} entries based on review feedback and evidence."
            }
        else:
            # Use the original non-threaded function
            improvement_result = improve_keech_entries(time_entries, review_feedback, filtered_evidence)
        
        # Get AI model and provider that were used
        ai_model = os.getenv('AI_MODEL', 'None specified')
        ai_provider = os.getenv('AI_PROVIDER', 'openai')
        
        # Build the response
        response_data = {
            "improvement_result": improvement_result,
            "ai_info": {
                "provider": ai_provider,
                "model": ai_model,
                "openai_key_available": bool(llm_client.openai_api_key),
                "anthropic_key_available": bool(llm_client.anthropic_api_key),
                "threading_used": use_threading,
                "max_workers": max_workers if use_threading else 0
            },
            "summary": {
                "total_entries": len(time_entries),
                "period_type": period_type,
                "selected_period": selected_period,
                "case_evidence_used": filtered_evidence is not None,
                "evidence_length": len(filtered_evidence) if filtered_evidence else 0
            }
        }
        
        logger.info("Keech improvement completed successfully")
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error improving Keech entries: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Handle timeout errors specially
        error_message = str(e)
        if "timed out" in error_message.lower():
            return jsonify({
                "error": "The AI service timed out. This usually means the model is too busy or the input is too complex. Please try a different AI model or try again later.",
                "error_details": error_message,
                "error_type": "timeout"
            }), 504  # Gateway Timeout status code
        
        return jsonify({"error": str(e)}), 500

# Add a new route to process multiple periods at once
@app.route('/api/process-multiple-periods', methods=['POST'])
def process_multiple_periods():
    """
    Endpoint to process multiple periods in parallel
    """
    logger.info("Received multi-period processing request")
    try:
        data = request.json
        file_path = data.get('file_path')
        period_type = data.get('period_type', 'weekly')
        periods = data.get('periods', [])  # List of periods to process
        custom_instructions = data.get('custom_instructions', '')
        evidence_path = data.get('evidence_path')
        ai_provider = data.get('ai_provider')
        ai_model = data.get('ai_model')
        max_workers = data.get('max_workers', 3)  # Default to 3 for parallel period processing
        
        # Set environment variables for the AI service if provided
        if ai_provider:
            os.environ['AI_PROVIDER'] = ai_provider
            logger.info(f"Setting AI provider to: {ai_provider}")
        if ai_model:
            os.environ['AI_MODEL'] = ai_model
            logger.info(f"Setting AI model to: {ai_model}")
        
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return jsonify({"error": "File not found"}), 404
            
        if not periods:
            logger.warning("No periods provided")
            return jsonify({"error": "No periods provided for processing"}), 400
        
        # Process a single period
        def process_period(period):
            try:
                logger.info(f"Processing period: {period}")
                
                # Step 1: Review the entries for this period
                review_response = review_keech_entries_for_period(
                    file_path=file_path,
                    period_type=period_type,
                    selected_period=period,
                    custom_instructions=custom_instructions,
                    ai_provider=ai_provider,
                    ai_model=ai_model,
                    use_threading=True  # Always use threading for the review step
                )
                
                if not review_response or not review_response.get('full_response'):
                    raise Exception(f"Failed to review period {period}")
                
                # Step 2: Get period-specific evidence
                filtered_evidence = None
                if evidence_path and os.path.exists(evidence_path):
                    filtered_evidence = get_evidence_for_period(evidence_path, period)
                
                # Step 3: Improve the entries based on the review
                time_entries = get_entries_for_period(file_path, period_type, period)
                
                improvement_result = improve_keech_entries(
                    time_entries, 
                    review_response.get('full_response'),
                    filtered_evidence
                )
                
                return {
                    "period": period,
                    "review": review_response,
                    "improvements": improvement_result,
                    "status": "success"
                }
            except Exception as e:
                logger.error(f"Error processing period {period}: {str(e)}")
                return {
                    "period": period,
                    "error": str(e),
                    "status": "error"
                }
        
        # Process periods in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks for each period
            future_to_period = {executor.submit(process_period, period): period for period in periods}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_period):
                period = future_to_period[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed processing for period {period}")
                except Exception as e:
                    logger.error(f"Error in future for period {period}: {str(e)}")
                    results.append({
                        "period": period,
                        "error": str(e),
                        "status": "error"
                    })
        
        # Sort results by period for consistency
        results.sort(key=lambda x: x.get('period', ''))
        
        # Build the response
        response_data = {
            "results": results,
            "summary": {
                "total_periods": len(periods),
                "successful_periods": sum(1 for r in results if r.get('status') == 'success'),
                "failed_periods": sum(1 for r in results if r.get('status') == 'error'),
                "period_type": period_type
            },
            "ai_info": {
                "provider": ai_provider,
                "model": ai_model,
                "max_workers": max_workers
            }
        }
        
        logger.info(f"Multi-period processing completed. Processed {len(periods)} periods.")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in multi-period processing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Helper function to get entries for a specific period
def get_entries_for_period(file_path, period_type, selected_period):
    """Get time entries for a specific period"""
    processed_data = process_timesheet(file_path, period_type)
    
    # Filter to the specified period
    filtered_data = [
        period_data for period_data in processed_data['detailed_data'] 
        if period_data.get('period') == selected_period
    ]
    
    # Extract entries
    entries = []
    for period_data in filtered_data:
        entries.extend(period_data.get('entries', []))
    
    return entries

# Helper function to review entries for a specific period
def review_keech_entries_for_period(
    file_path, period_type, selected_period, 
    custom_instructions, ai_provider, ai_model, 
    use_threading=True
):
    """Review time entries for a specific period"""
    # Get entries for the period
    entries = get_entries_for_period(file_path, period_type, selected_period)
    
    if not entries:
        return {"status": "error", "error": f"No entries found for period {selected_period}"}
    
    # Call the appropriate review function
    if use_threading and len(entries) > 10:
        from services.threaded_ai_service import threaded_batch_review_entries
        return threaded_batch_review_entries(
            entries, 
            custom_instructions,
            max_workers=5,
            batch_size=20
        )
    else:
        from services.ai_service import review_keech_entries
        return review_keech_entries(entries, custom_instructions)
        
@app.route('/api/download-improved-keech-entries', methods=['POST'])
def download_improved_keech_entries():
    """
    Endpoint to download improved Keech entries as a CSV file
    """
    logger.info("Received download improved Keech entries request")
    try:
        data = request.json
        csv_content = data.get('csv_content')
        
        if not csv_content:
            logger.warning("No CSV content provided")
            return jsonify({"error": "No CSV content provided"}), 400
        
        # Create temporary file with the CSV content
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        
        # Write the CSV content directly to the file
        with open(temp_file.name, 'w', encoding='utf-8') as f:
            f.write(csv_content)
        
        filename = "improved_keech_entries.csv"
        
        logger.info(f"Prepared download file: {filename}")
        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name=filename,
            mimetype='text/csv'
        )
    except Exception as e:
        logger.error(f"Error downloading improved Keech entries: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500
        
@app.route('/api/download/<file_type>', methods=['POST'])
def download_file(file_type):
    """
    Endpoint to download generated files
    file_type can be 'keech', 'redacted', or 'discount'
    """
    logger.info(f"Received download request for {file_type}")
    try:
        data = request.json
        content = data.get('content')
        
        if not content:
            logger.warning("No content provided for download")
            return jsonify({"error": "No content provided"}), 400
        
        # Log content structure to understand what we're working with
        logger.info(f"Content type: {type(content)}")
        if isinstance(content, dict):
            logger.info(f"Content keys: {list(content.keys())}")
        elif isinstance(content, list):
            logger.info(f"Content length: {len(content)}")
            if content and isinstance(content[0], dict):
                logger.info(f"First item keys: {list(content[0].keys())}")
        
        # Create temporary file with the content
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        
        if file_type == 'keech':
            # For Keech declaration, flatten the structure
            if isinstance(content, dict) and 'periods' in content:
                # Extract all entries from all periods
                all_entries = []
                for period in content.get('periods', []):
                    if 'entries' in period:
                        all_entries.extend(period['entries'])
                
                # Convert to DataFrame
                if all_entries:
                    pd.DataFrame(all_entries).to_csv(temp_file.name, index=False)
                else:
                    pd.DataFrame().to_csv(temp_file.name, index=False)
            else:
                # Try direct conversion
                pd.DataFrame(content).to_csv(temp_file.name, index=False)
            
            filename = "keech_declaration.csv"
        
        elif file_type == 'redacted':
            # For redacted timesheet, compile all entries from all periods
            all_entries = []
            
            if isinstance(content, dict):
                # Extract from detailed_data structure if available
                if 'detailed_data' in content:
                    for period in content.get('detailed_data', []):
                        if 'entries' in period:
                            all_entries.extend(period['entries'])
                # Otherwise try raw_data
                elif 'raw_data' in content:
                    all_entries = content['raw_data']
            elif isinstance(content, list):
                # Direct list of entries
                all_entries = content
            
            # Convert to DataFrame
            if all_entries:
                pd.DataFrame(all_entries).to_csv(temp_file.name, index=False)
            else:
                logger.warning("No entries found in redacted content")
                pd.DataFrame().to_csv(temp_file.name, index=False)
            
            filename = "redacted_timesheet.csv"
            
        elif file_type == 'discount':
            # For discount analysis, compile all entries from all periods
            all_entries = []
            summary_data = None
            
            if isinstance(content, dict):
                # Extract summary if available
                if 'discount_summary' in content:
                    summary_data = content['discount_summary']
                
                # Extract from detailed_data structure if available
                if 'detailed_data' in content:
                    for period in content.get('detailed_data', []):
                        if 'entries' in period:
                            all_entries.extend(period['entries'])
            elif isinstance(content, list):
                # Direct list of entries
                all_entries = content
            
            # Create DataFrame from entries
            if all_entries:
                # Filter to include relevant discount fields
                df = pd.DataFrame(all_entries)
                
                # Add summary as extra rows if available
                if summary_data and isinstance(summary_data, dict):
                    # Create a summary row at the end
                    summary_df = pd.DataFrame([{
                        'Description': '===== DISCOUNT SUMMARY =====',
                        'OriginalHours': summary_data.get('original_hours', 0),
                        'DiscountedHours': summary_data.get('discounted_hours', 0),
                        'HoursSaved': summary_data.get('hours_saved', 0),
                        'PercentageSaved': summary_data.get('percentage_saved', 0),
                        'EntriesDiscounted': summary_data.get('entries_discounted', 0)
                    }])
                    
                    # Append to the main dataframe
                    df = pd.concat([df, summary_df], ignore_index=True)
                
                df.to_csv(temp_file.name, index=False)
            else:
                logger.warning("No entries found in discount content")
                pd.DataFrame().to_csv(temp_file.name, index=False)
            
            filename = "discount_analysis.csv"
        
        else:
            logger.warning(f"Invalid file type: {file_type}")
            return jsonify({"error": "Invalid file type"}), 400
        
        logger.info(f"Prepared download file: {filename}")
        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name=filename,
            mimetype='text/csv'
        )
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

# Simple test route to verify the server is running
@app.route('/api/test', methods=['GET'])
def test_api():
    logger.info("Test API endpoint called")
    return jsonify({"status": "success", "message": "API is running"}), 200

@app.route('/test', methods=['GET'])
def test_root():
    logger.info("Root test endpoint called")
    return jsonify({"status": "success", "message": "API is running at root"}), 200
    
@app.route('/api/health', methods=['GET'])
def health_check():
    logger.info("Health check endpoint called")
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "app": "timesheet-analyzer",
        "version": "1.0.0"
    })

@app.route('/api/test-ai', methods=['GET'])
def test_ai_connection():
    """
    Simple endpoint to test AI connection
    """
    from services.ai_service import llm_client
    
    logger.info("Testing AI connection")
    try:
        # Log API key status
        openai_key_available = bool(llm_client.openai_api_key)
        anthropic_key_available = bool(llm_client.anthropic_api_key)
        
        logger.info(f"OpenAI API key available: {openai_key_available}")
        logger.info(f"Anthropic API key available: {anthropic_key_available}")
        
        # Get available models
        models = llm_client.get_flat_model_list()
        
        # Test a simple completion if we have a key
        test_result = None
        ai_provider = None
        ai_model = None
        
        if openai_key_available:
            logger.info("Testing OpenAI connection")
            ai_provider = 'openai'
            ai_model = 'gpt-3.5-turbo'  # Use a reliable model for testing
            
            # Test the model
            test_result = llm_client.generate_text(
                model_id=ai_model,
                provider=ai_provider,
                prompt="Say 'Hello, I am an AI assistant!' and nothing else.",
                temperature=0.1
            )
            logger.info(f"OpenAI test result: {test_result}")
            
        elif anthropic_key_available:
            logger.info("Testing Anthropic connection")
            ai_provider = 'anthropic'
            ai_model = 'claude-3-haiku-20240307'  # Use a reliable model for testing
            
            # Test the model
            test_result = llm_client.generate_text(
                model_id=ai_model,
                provider=ai_provider,
                prompt="Say 'Hello, I am an AI assistant!' and nothing else.",
                temperature=0.1
            )
            logger.info(f"Anthropic test result: {test_result}")
                
        return jsonify({
            "openai_api_key_available": openai_key_available,
            "anthropic_api_key_available": anthropic_key_available,
            "models_found": len(models),
            "test_result": test_result,
            "test_provider": ai_provider,
            "test_model": ai_model
        })
    except Exception as e:
        logger.error(f"Error testing AI connection: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/api/set-api-keys', methods=['POST'])
def set_api_keys():
    """
    Endpoint to set API keys for AI services
    """
    logger.info("Setting API keys")
    try:
        data = request.json
        openai_key = data.get('openai_key')
        anthropic_key = data.get('anthropic_key')
        
        # Set environment variables
        if openai_key:
            os.environ['OPENAI_API_KEY'] = openai_key
            logger.info("OpenAI API key set")
        
        if anthropic_key:
            os.environ['ANTHROPIC_API_KEY'] = anthropic_key
            logger.info("Anthropic API key set")
            
        # Update the LLM client with the new API keys
        from services.ai_service import llm_client
        if openai_key or anthropic_key:
            llm_client.update_api_keys(openai_key=openai_key, anthropic_key=anthropic_key)
            logger.info("LLM client updated with new API keys")
        
        return jsonify({
            "success": True,
            "message": "API keys updated successfully",
            "openai_key_set": bool(openai_key),
            "anthropic_key_set": bool(anthropic_key)
        })
    except Exception as e:
        logger.error(f"Error setting API keys: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/ai-models', methods=['GET'])
def get_ai_models():
    """
    Endpoint to get the available AI models
    """
    logger.info("Retrieving available AI models")
    try:
        # Force refresh of models to get the latest available
        refresh = request.args.get('refresh', 'false').lower() == 'true'
        
        # Get the available models
        models = llm_client.get_flat_model_list()
        
        # Get current model and provider settings
        current_model = os.getenv('AI_MODEL')
        current_provider = os.getenv('AI_PROVIDER', 'openai')
        
        return jsonify({
            "models": models,
            "current_settings": {
                "model": current_model,
                "provider": current_provider
            },
            "providers": list(llm_client.available_models.keys())
        })
    except Exception as e:
        logger.error(f"Error retrieving AI models: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/test-redaction', methods=['POST'])
def test_redaction():
    """
    Endpoint to test redaction on a specific set of timesheet entries
    
    This allows you to test different AI providers, models, and custom instructions
    on a specific timesheet section to see how the redaction works.
    """
    logger.info("Received test redaction request")
    try:
        data = request.json
        entries = data.get('entries', [])
        ai_provider = data.get('ai_provider')
        ai_model = data.get('ai_model')
        custom_instructions = data.get('custom_instructions', '')
        
        if not entries:
            logger.warning("No entries provided for redaction test")
            return jsonify({"error": "No entries provided for redaction test"}), 400
        
        # Set environment variables for the AI service if provided
        if ai_provider:
            os.environ['AI_PROVIDER'] = ai_provider
            logger.info(f"Setting AI provider to: {ai_provider}")
        if ai_model:
            os.environ['AI_MODEL'] = ai_model
            logger.info(f"Setting AI model to: {ai_model}")
        
        logger.info(f"Testing redaction with {len(entries)} entries")
        
        # Prepare data for AI in the same format that ai_redact_timesheet expects
        processed_data = {
            'detailed_data': [
                {
                    'period': 'test-period',
                    'entries': entries
                }
            ],
            'raw_data': entries  # For simplicity, use the same entries as raw data
        }
        
        # Create redacted version
        from services.ai_service import ai_redact_timesheet
        redacted_data = ai_redact_timesheet(processed_data, custom_instructions)
        
        # Get AI model and provider that were used
        ai_model_used = os.getenv('AI_MODEL', 'None specified')
        ai_provider_used = os.getenv('AI_PROVIDER', 'openai')
        
        return jsonify({
            "original_entries": entries,
            "redacted_entries": redacted_data['detailed_data'][0]['entries'] if redacted_data and 'detailed_data' in redacted_data and redacted_data['detailed_data'] else [],
            "ai_info": {
                "provider": ai_provider_used,
                "model": ai_model_used,
                "custom_instructions": custom_instructions
            }
        })
        
    except Exception as e:
        logger.error(f"Error in redaction test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Handle timeout errors specially
        error_message = str(e)
        if "timed out" in error_message.lower():
            return jsonify({
                "error": "The AI service timed out. This usually means the model is too busy or the input is too complex. Please try a different AI model or try again later.",
                "error_details": error_message,
                "error_type": "timeout"
            }), 504  # Gateway Timeout status code
        
        # Handle other errors
        return jsonify({"error": error_message}), 500

@app.route('/api/analyze-with-ai', methods=['POST'])
def analyze_with_ai():
    """
    Endpoint to analyze timesheet sections using the AI
    
    This endpoint takes timesheet sections, sends them to the selected AI model
    with provided instructions, and returns the AI's analysis for each section.
    """
    logger.info("Received AI analysis request")
    try:
        data = request.json
        sections = data.get('sections', [])
        instructions = data.get('instructions', '')
        ai_provider = data.get('ai_provider')
        ai_model = data.get('ai_model')
        
        if not sections:
            logger.warning("No sections provided for analysis")
            return jsonify({"error": "No sections provided for analysis"}), 400
        
        # Set environment variables for the AI service if provided
        if ai_provider:
            os.environ['AI_PROVIDER'] = ai_provider
            logger.info(f"Setting AI provider to: {ai_provider}")
        if ai_model:
            os.environ['AI_MODEL'] = ai_model
            logger.info(f"Setting AI model to: {ai_model}")
        
        # Create system prompt with the instructions
        system_prompt = f"""
        You are an expert legal assistant specializing in analyzing legal timesheets.
        Your task is to analyze each timesheet section according to the following instructions:
        
        {instructions}
        
        Provide a detailed analysis for each timesheet section, focusing on the specific requirements
        in the instructions above. Be concise but thorough.
        """
        
        # Process each section with the AI
        results = []
        for i, section in enumerate(sections):
            logger.info(f"Processing section {i+1} of {len(sections)}")
            try:
                # Prepare the section data for the AI
                section_text = json.dumps(section, indent=2)
                prompt = f"Please analyze the following timesheet section:\n\n{section_text}"
                
                # Call the AI
                response = llm_client.generate_text(
                    model_id=ai_model,
                    provider=ai_provider,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.3,  # Lower temperature for more consistent results
                    max_tokens=2000
                )
                
                # Add the result
                results.append({
                    "section_id": i,
                    "original_section": section,
                    "analysis": response
                })
                
            except Exception as section_error:
                logger.error(f"Error processing section {i+1}: {str(section_error)}")
                results.append({
                    "section_id": i,
                    "original_section": section,
                    "error": str(section_error)
                })
        
        logger.info(f"Completed AI analysis of {len(sections)} sections")
        return jsonify({
            "results": results,
            "ai_used": {
                "provider": ai_provider,
                "model": ai_model
            }
        })
        
    except Exception as e:
        logger.error(f"Error during AI analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/review-keech-entries', methods=['POST'])
def review_keech_entries_api():
    """
    Endpoint to review time entries for a Keech Declaration using threading
    """
    logger.info("Received Keech review request")
    try:
        data = request.json
        file_path = data.get('file_path')
        period_type = data.get('period_type', 'weekly')
        selected_period = data.get('selected_period')
        custom_instructions = data.get('custom_instructions', '')
        ai_provider = data.get('ai_provider')
        ai_model = data.get('ai_model')
        use_threading = data.get('use_threading', True)  # New parameter to enable/disable threading
        max_workers = data.get('max_workers', 3)  # Number of concurrent threads
        batch_size = data.get('batch_size', 20)  # Entries per batch for batch processing
        
        # Set environment variables for the AI service if provided
        if ai_provider:
            os.environ['AI_PROVIDER'] = ai_provider
            logger.info(f"Setting AI provider to: {ai_provider}")
        if ai_model:
            os.environ['AI_MODEL'] = ai_model
            logger.info(f"Setting AI model to: {ai_model}")
        
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return jsonify({"error": "File not found"}), 404
            
        # Process the timesheet to get the time entries
        processed_data = process_timesheet(file_path, period_type)
        
        # If a specific period was selected, filter the data
        if selected_period:
            logger.info(f"Filtering data to only include period: {selected_period}")
            
            # Filter detailed_data
            filtered_detailed_data = [
                period_data for period_data in processed_data['detailed_data'] 
                if period_data.get('period') == selected_period
            ]
            
            if not filtered_detailed_data:
                logger.warning(f"No data found for period: {selected_period}")
                return jsonify({"error": f"No data found for period: {selected_period}"}), 404
                
            # Extract entries from the filtered period
            time_entries = []
            for period_data in filtered_detailed_data:
                time_entries.extend(period_data.get('entries', []))
        else:
            # Get all entries from all periods
            time_entries = []
            for period_data in processed_data['detailed_data']:
                time_entries.extend(period_data.get('entries', []))
        
        if not time_entries:
            logger.warning("No time entries found for review")
            return jsonify({"error": "No time entries found for review"}), 400
            
        logger.info(f"Reviewing {len(time_entries)} time entries")
        
        # Call the appropriate review function based on threading preference
        if use_threading and len(time_entries) > batch_size:
            logger.info(f"Using threaded batch review with {max_workers} workers and batch size {batch_size}")
            review_result = threaded_batch_review_entries(
                time_entries, 
                custom_instructions,
                max_workers=max_workers,
                batch_size=batch_size
            )
        else:
            # Use the original non-threaded function
            review_result = review_keech_entries(time_entries, custom_instructions)
        
        # Get AI model and provider that were used
        ai_model = os.getenv('AI_MODEL', 'None specified')
        ai_provider = os.getenv('AI_PROVIDER', 'openai')
        
        # Build the response
        response_data = {
            "review_result": review_result,
            "ai_info": {
                "provider": ai_provider,
                "model": ai_model,
                "openai_key_available": bool(llm_client.openai_api_key),
                "anthropic_key_available": bool(llm_client.anthropic_api_key),
                "threading_used": use_threading,
                "max_workers": max_workers if use_threading else 0,
                "batch_size": batch_size if use_threading else 0
            },
            "summary": {
                "total_entries": len(time_entries),
                "period_type": period_type,
                "selected_period": selected_period
            }
        }
        
        logger.info("Keech review completed successfully")
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error reviewing Keech entries: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Handle timeout errors specially
        error_message = str(e)
        if "timed out" in error_message.lower():
            return jsonify({
                "error": "The AI service timed out. This usually means the model is too busy or the input is too complex. Please try a different AI model or try again later.",
                "error_details": error_message,
                "error_type": "timeout"
            }), 504  # Gateway Timeout status code
        
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask development server...")
    app.run(debug=True, host='0.0.0.0', port=5001)
    logger.info("Server has shut down.")