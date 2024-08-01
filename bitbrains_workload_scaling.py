#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pyarrow as pa
from datetime import datetime as datetime
import sys
import math

df_meta = pd.read_parquet(f"/Users/didepoyraz/Desktop/honours/Demo/traces_all/bitbrains-small/trace/meta.parquet")
df_trace = pd.read_parquet(f"/Users/didepoyraz/Desktop/honours/Demo/traces_all/bitbrains-small/trace/trace.parquet")

scaling_dict = {}
job_completion = []
job_completion_average = []
index = 0

new_rows_trace = []
new_rows_meta = []

squashed_area_trace = 0
squashed_area_system = 0
index = 0 

aggregated_trace = df_trace.groupby('id').agg({
    'timestamp': 'first',  
    'duration': 'sum',     
    'cpu_count': 'first',    
    'cpu_usage': 'mean'     
}).reset_index()

def add_duplicate_row(server_id, row, cpu_usage, df_meta_result, cpu_speed_mhz):
    global squashed_area_system
    
    index = scaling_dict[server_id][1] 
    new_id = f"{server_id}-{index}"
    scaling_dict[server_id][1] += 1
    
    modified_row = row.copy()
    modified_row['id'] = new_id 
    modified_row['cpu_usage'] = cpu_usage 
    modified_row['cpu_count'] = math.ceil(row['cpu_usage'] / cpu_speed_mhz)
    
    meta_row = df_meta_result.loc[df_meta['id'] == server_id].copy()
    meta_row['id'] = new_id

    new_rows_meta.append(meta_row.iloc[0])
    new_rows_trace.append(modified_row)
    
    return
    
def is_fractional(number):
    if number != int(number):
        return number - int(number)
    else: 
        return 0

def adjust_meta(row, cpu_speed_mhz, cpu_count):
    final_scaling_factor = 1 

    system_cpu_capacity = cpu_speed_mhz * cpu_count
    final_scaling_factor = system_cpu_capacity / row['cpu_capacity']
    
    row['cpu_capacity'] = system_cpu_capacity
    row['cpu_count'] = cpu_count
    
    # (system CPU capacity * System CPU count) / (Workload CPU capacity )
    
    server_id = row['id']
    scaling_dict[server_id] = [final_scaling_factor, 1]

    return row

def adjust_trace(row, cpu_speed_mhz, cpu_count, df_meta_result):
    global squashed_area_trace, squashed_area_system, index
    
    server_id = row['id']
    final_factor = scaling_dict[server_id][0]
    
    total_system_capacity = cpu_speed_mhz * cpu_count
    task_total_cpu_usage = row['cpu_usage'] * row['duration']
    
    squashed_area_trace += task_total_cpu_usage    
        
    if (row['cpu_usage'] != 0) & (row['cpu_usage'] > total_system_capacity):    
        row['cpu_usage'] *= final_factor
        row['cpu_count'] = math.ceil(row['cpu_usage'] / cpu_speed_mhz)
        
        job_completion_ratio = (row['cpu_usage']*row['duration'] / task_total_cpu_usage) 
        
        if job_completion_ratio < 1:
            index += 1
            leftover_work = task_total_cpu_usage * (1 - (job_completion_ratio))
            leftover_cpu_usage = leftover_work / row['duration']
            
            num_rows_to_add = 1
            if leftover_cpu_usage > total_system_capacity:

                num_rows_to_add = leftover_cpu_usage / total_system_capacity
                leftover_cpu_usage = total_system_capacity
            
                fraction = is_fractional(num_rows_to_add)
                if fraction: 
                    fractional_cpu_usage = fraction * (total_system_capacity)
                    add_duplicate_row(server_id,row, fractional_cpu_usage, df_meta_result, cpu_speed_mhz)
                    squashed_area_system += row['duration'] * fractional_cpu_usage
         
            for _ in range(int(num_rows_to_add)):
                add_duplicate_row(server_id,row, leftover_cpu_usage, df_meta_result, cpu_speed_mhz)
                squashed_area_system += row['duration'] * leftover_cpu_usage 
                
        job_completion_percentage = ((row['cpu_usage']* row['duration']) / task_total_cpu_usage) * 100
        job_completion.append(job_completion_percentage)
        
    else:
        row['cpu_count'] = cpu_count
    squashed_area_system += (row['cpu_usage']* row['duration'])
    return row

def save_average_job_completion():
    total_completion_rate = 0
    num_items = len(job_completion)

    for value in job_completion:

        total_completion_rate += value

    average_completion_rate = total_completion_rate / num_items
    job_completion_average.append(average_completion_rate)

def correct_schema_meta(df_meta_adjusted):
   
    df_meta_adjusted['cpu_count'] = df_meta_adjusted['cpu_count'].astype('int32')
    df_meta_adjusted['cpu_capacity'] = df_meta_adjusted['cpu_capacity'].astype('float64')
    df_meta_adjusted['mem_capacity'] = df_meta_adjusted['mem_capacity'].astype('int64')

    schema = pa.schema([
        pa.field('id', pa.string(), nullable=False),
        pa.field('start_time', pa.timestamp('ms', tz='UTC'), nullable=False),
        pa.field('stop_time', pa.timestamp('ms', tz='UTC'), nullable=False),
        pa.field('cpu_count', pa.int32(), nullable=False),
        pa.field('cpu_capacity', pa.float64(), nullable=False),
        pa.field('mem_capacity', pa.int64(), nullable=False)
    ])

    table = pa.Table.from_pandas(df_meta_adjusted, schema=schema)
    
    #change the file path to where you want to store the scaled workload
    pq.write_table(table, '/Users/didepoyraz/Desktop/honours/Demo/traces/bitbrains-small/trace/meta.parquet')
    
def correct_schema_trace(df_trace_adjusted):
 
    df_trace_adjusted['duration'] = df_trace_adjusted['duration'].astype('int64')
    df_trace_adjusted['cpu_count'] = df_trace_adjusted['cpu_count'].astype('int32')
    df_trace_adjusted['cpu_usage'] = df_trace_adjusted['cpu_usage'].astype('float64')

    schema = pa.schema([
        pa.field('id', pa.string(), nullable=False),
        pa.field('timestamp', pa.timestamp('ms', tz='UTC'), nullable=False),
        pa.field('duration', pa.int64(), nullable=False),
        pa.field('cpu_count', pa.int32(), nullable=False),
        pa.field('cpu_usage', pa.float64(), nullable=False)
    ])

    table = pa.Table.from_pandas(df_trace_adjusted, schema=schema)

    #change the file path to where you want to store the scaled workload
    pq.write_table(table,'/Users/didepoyraz/Desktop/honours/Demo/traces/bitbrains-small/trace/trace.parquet')

def duplicate_dataframes(df_trace, df_meta, n_duplicates):

    df_trace['id'] = df_trace['id'].astype(int)
    df_meta['id'] = df_meta['id'].astype(int)
    
    df_trace_combined = df_trace.copy()
    df_meta_combined = df_meta.copy()
    
    last_id = df_trace['id'].max()
    
    for i in range(n_duplicates):

        df_trace_duplicate = df_trace.copy()
        df_meta_duplicate = df_meta.copy()
        
        df_trace_duplicate['id'] = df_trace_duplicate['id'] + (last_id + 1) * (i + 1)
        df_meta_duplicate['id'] = df_meta_duplicate['id'] + (last_id + 1) * (i + 1)
        
        df_trace_combined = pd.concat([df_trace_combined, df_trace_duplicate], ignore_index=True)
        df_meta_combined = pd.concat([df_meta_combined, df_meta_duplicate], ignore_index=True)
    
    df_trace_combined['id'] = df_trace_combined['id'].astype(str)
    df_meta_combined['id'] = df_meta_combined['id'].astype(str)
    
    return df_trace_combined, df_meta_combined

def main(cpu_speed_mhz, cpu_count):
    global new_rows_trace, new_rows_meta 
    
    df_meta_adjusted = df_meta.apply(adjust_meta, axis=1, args=(cpu_speed_mhz, cpu_count))
    df_trace_adjusted = aggregated_trace.apply(adjust_trace, axis=1, args=(cpu_speed_mhz, cpu_count, df_meta_adjusted))

    if new_rows_trace and new_rows_meta: 
        new_df_trace = pd.DataFrame(new_rows_trace)
        df_trace_adjusted = pd.concat([df_trace_adjusted, new_df_trace], ignore_index=True)

        new_df_meta = pd.DataFrame(new_rows_meta)
        df_meta_adjusted = pd.concat([df_meta_adjusted, new_df_meta], ignore_index=True)

    print(f"Squashed area of trace: {squashed_area_trace}, and the squashed area of system: {squashed_area_system}, tasks duplicated {index} times")

    correct_schema_meta(df_meta_adjusted)
    correct_schema_trace(df_trace_adjusted)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: bitbrains_workload_scaling.py <cpu_speed_mhz> <cpu_count>")
        sys.exit(1)
        
    cpu_speed_mhz = int(sys.argv[1])
    cpu_count = int(sys.argv[2])

    main(cpu_speed_mhz, cpu_count)

