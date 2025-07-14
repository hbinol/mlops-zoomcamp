#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
from datetime import datetime

# Set environment variables for VS Code compatibility
os.environ['S3_ENDPOINT_URL'] = 'http://localhost:4566'
os.environ['INPUT_FILE_PATTERN'] = 's3://nyc-duration/in/{year:04d}-{month:02d}.parquet'
os.environ['OUTPUT_FILE_PATTERN'] = 's3://nyc-duration/out/{year:04d}-{month:02d}.parquet'
os.environ['AWS_ACCESS_KEY_ID'] = 'test'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'test'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

from batch import get_input_path


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def create_test_data():
    """Create the same test dataframe used in Q3 unit test"""
    data = [
        (None, None, dt(1, 1), dt(1, 10)),      # None values, 9 minute duration
        (1, 1, dt(1, 2), dt(1, 10)),           # Valid values, 8 minute duration  
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),  # One None value, 59 second duration (< 1 minute)
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      # Valid values, 1 hour 1 second duration (> 60 minutes)
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df_input = pd.DataFrame(data, columns=columns)
    
    return df_input


def get_s3_options():
    """Get S3 storage options with proper credentials for Localstack"""
    options = {}
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL')
    
    if s3_endpoint_url:
        # For Localstack, we need to provide AWS credentials (even dummy ones)
        options = {
            'client_kwargs': {
                'endpoint_url': s3_endpoint_url,
                'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID', 'test'),
                'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY', 'test'),
                'region_name': os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
            }
        }
    
    return options, s3_endpoint_url


def main():
    print("🧪 End-to-End Integration Test")
    print("=" * 50)
    print("Step 1: Save test data to S3")
    
    # Create test dataframe
    df_input = create_test_data()
    print(f"✅ Created test dataframe with {len(df_input)} rows")
    print(f"   Columns: {list(df_input.columns)}")
    
    # Get input file path for January 2023
    year, month = 2023, 1
    input_file = get_input_path(year, month)
    print(f"📁 Target file: {input_file}")
    
    # Set up S3 options with credentials
    options, s3_endpoint_url = get_s3_options()
    
    if s3_endpoint_url:
        print(f"🔗 Using S3 endpoint: {s3_endpoint_url}")
        print(f"🔑 AWS Access Key: {options['client_kwargs'].get('aws_access_key_id', 'NOT SET')}")
    else:
        print("🌐 Using default AWS S3")
    
    # Save dataframe to S3
    try:
        print("💾 Saving dataframe to S3...")
        df_input.to_parquet(
            input_file,
            engine='pyarrow',
            compression=None,
            index=False,
            storage_options=options
        )
        print("✅ Successfully saved test data to S3!")
        
    except Exception as e:
        print(f"❌ Error saving to S3: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure Localstack is running: docker-compose ps")
        print("   2. Check environment variables:")
        print(f"      S3_ENDPOINT_URL: {os.getenv('S3_ENDPOINT_URL', 'NOT SET')}")
        print(f"      AWS_ACCESS_KEY_ID: {os.getenv('AWS_ACCESS_KEY_ID', 'NOT SET')}")
        print(f"      AWS_SECRET_ACCESS_KEY: {os.getenv('AWS_SECRET_ACCESS_KEY', 'NOT SET')}")
        print("   3. Try setting: export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test")
        return False
    
    print("\n🔍 Verification:")
    print(f"   File saved to: {input_file}")
    print(f"   Rows: {len(df_input)}")
    print(f"   Data preview:")
    print(df_input.head())
    
    print("\n🚀 Test data saved successfully!")
    
    # Step 2: Run the batch.py script for January 2023
    print("\n" + "=" * 50)
    print("🔄 Running batch.py script for January 2023...")
    
    batch_command = "pipenv run python batch.py 2023 1"
    print(f"Command: {batch_command}")
    
    exit_code = os.system(batch_command)
    
    if exit_code != 0:
        print(f"❌ Batch script failed with exit code: {exit_code}")
        return False
    
    print("✅ Batch script completed successfully!")
    
    # Step 3: Read and verify the results
    print("\n" + "=" * 50)
    print("🔍 Verifying results...")
    
    # Import after setting environment variables
    from batch import get_output_path, read_data
    
    output_file = get_output_path(year, month)
    print(f"📁 Reading results from: {output_file}")
    
    try:
        # Read the output file (note: we don't need categorical for result data)
        # But we'll use read_data's S3 functionality by passing empty categorical list
        df_result = pd.read_parquet(output_file, storage_options=get_s3_options()[0])
        
        print("✅ Successfully read results from S3!")
        print(f"📊 Result shape: {df_result.shape}")
        print("📋 Result columns:", list(df_result.columns))
        print("📋 First few predictions:")
        print(df_result.head())
        
        # Verify the results
        expected_columns = ['ride_id', 'predicted_duration']
        if list(df_result.columns) == expected_columns:
            print("✅ Columns are correct!")
        else:
            print(f"❌ Unexpected columns. Expected {expected_columns}, got {list(df_result.columns)}")
        
        if len(df_result) > 0:
            print(f"✅ Results contain {len(df_result)} predictions")
            print(f"🎯 Mean predicted duration: {df_result['predicted_duration'].mean():.2f} minutes")
        else:
            print("❌ No predictions found in results")
        
    except Exception as e:
        print(f"❌ Error reading results: {e}")
        return False
    
    print("\n🎉 Complete end-to-end integration test successful!")
    print("\nTo verify with AWS CLI:")
    if s3_endpoint_url:
        print(f"   Input:  aws s3 ls s3://nyc-duration/in/ --endpoint-url={s3_endpoint_url}")
        print(f"   Output: aws s3 ls s3://nyc-duration/out/ --endpoint-url={s3_endpoint_url}")
    else:
        print("   Input:  aws s3 ls s3://nyc-duration/in/")
        print("   Output: aws s3 ls s3://nyc-duration/out/")
    
    return True


if __name__ == "__main__":
    main() 