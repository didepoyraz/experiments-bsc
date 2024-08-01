#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pyarrow as pa
from datetime import datetime as datetime
import math 
import sys

df_trace = pd.read_parquet(f"traces_all/2022-10-07_2022-10-14/trace/trace.parquet")

df_meta = pd.read_parquet(f"traces_all/2022-10-07_2022-10-14/trace/meta.parquet")

df_energy = pd.read_parquet(f"traces_all/surfsara_trace/trace/energy.parquet")

df_trace_small = pd.read_parquet(f"traces_all/surfsara_trace/trace/meta.parquet")
df_meta_small = pd.read_parquet(f"traces_all/surfsara_trace/trace/trace.parquet")

trace_cpu_count = 16
trace_cpu_core_speed = 2100

scaling_dict = {}
job_completion = []

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

system_cpu_capacity_per_node = None
trace_capacity_per_node = trace_cpu_count * trace_cpu_core_speed
final_scaling_factor = None

def add_duplicate_row(server_id, row, cpu_usage, total_duration_available, df_meta_result, cpu_speed_comp_mhz):
    global squashed_area_system
    index = scaling_dict[server_id][1]
    new_id = f"{server_id}-{index}"

    scaling_dict[server_id][1] += 1
    
    modified_row = row.copy()
    modified_row['id'] = new_id 
    modified_row['cpu_usage'] = cpu_usage 
    modified_row['cpu_count'] = math.ceil(row['cpu_usage'] / cpu_speed_comp_mhz)
    
    meta_row = df_meta_result.loc[df_meta['id'] == server_id].copy()

    meta_row['id'] = new_id

    new_rows_meta.append(meta_row.iloc[0])
    new_rows_trace.append(modified_row)
    squashed_area_system += total_duration_available * cpu_usage 
    
    return

def is_fractional(number):
    if number != int(number):
        return number - int(number)
    else: 
        return 0

# Adjusting Meta
def adjust_meta(row, cpu_speed_comp_mhz, cpu_count_comp):
    
    row['cpu_capacity'] = system_cpu_capacity_per_node
    row['cpu_count'] = cpu_count_comp
    
    # (system CPU capacity * System CPU count) / (Workload CPU capacity )
    
    start_time = row['start_time']
    stop_time = row['stop_time']
    
    duration_in_seconds = (stop_time - start_time).total_seconds()
    scaled_duration_in_seconds = duration_in_seconds / final_scaling_factor
    # if scaling factor is bigger than 1, the duration will get shorter,
    # if it is smaller than 1 the duration will get bigger
    
    if scaled_duration_in_seconds < 1:
        scaled_duration_in_seconds = 1
  
    adjusted_stop_time = start_time + pd.to_timedelta(scaled_duration_in_seconds, unit='s')
    
    trace_stop_time = df_meta['stop_time'].max()
    if adjusted_stop_time > trace_stop_time:
        adjusted_stop_time = trace_stop_time
    total_duration_in_ms = (trace_stop_time - pd.to_datetime(row['start_time'])).total_seconds() * 1000
  
    row['stop_time'] = adjusted_stop_time
    server_id = row['id']
    scaling_dict[server_id] = [total_duration_in_ms, 1]
 
    
    return row

# Adjusting Trace
def adjust_trace(row, cpu_speed_comp_mhz, cpu_count_comp, df_meta_result):
    global squashed_area_trace, squashed_area_system, index
    
    server_id = row['id']
    total_duration_available = scaling_dict[server_id][0]
    total_system_capacity = cpu_speed_comp_mhz * cpu_count_comp
    
    task_total_cpu_usage = row['cpu_usage'] * row['duration']
    squashed_area_trace += task_total_cpu_usage
    
    if row['cpu_usage'] != 0:      
        
        row['cpu_usage'] *= final_scaling_factor     
        row['duration'] /= final_scaling_factor 
        row['cpu_count'] = math.ceil(row['cpu_usage'] / cpu_speed_comp_mhz)
        
        if row['duration'] > total_duration_available:
            index += 1
            row['duration'] = total_duration_available
            job_completion_percentage = (row['cpu_usage']* row['duration']) / task_total_cpu_usage
            leftover_work = task_total_cpu_usage * (1 - (job_completion_percentage))
            leftover_cpu_usage = leftover_work / total_duration_available
            
            num_rows_to_add = 1
            if leftover_cpu_usage > (total_system_capacity):
               
                num_rows_to_add = leftover_cpu_usage / total_system_capacity
                leftover_cpu_usage = total_system_capacity
            
                fraction = is_fractional(num_rows_to_add)
                if fraction: 
                    fractional_cpu_usage = fraction * total_system_capacity
                    add_duplicate_row(server_id,row, fractional_cpu_usage, total_duration_available, df_meta_result, cpu_speed_comp_mhz)

            for _ in range(int(num_rows_to_add)):
                add_duplicate_row(server_id,row, leftover_cpu_usage, total_duration_available,df_meta_result, cpu_speed_comp_mhz)
                
        if row['duration'] < 1:
            # to make the cpu_usage in proportion to the 1 second, we multiply the cpu_usage with the duration to find the total work done and divide it by 1. 
            # because we know the duration will always be 1 we don't need to divide
            row['cpu_usage'] *= row['duration']
            row['duration'] = 1
            
        job_completion_percentage = ((row['cpu_usage']* row['duration']) / task_total_cpu_usage) * 100
        job_completion.append(job_completion_percentage)
        squashed_area_system += (row['cpu_usage']* row['duration'])
                  
    else:
        row['cpu_count'] = cpu_count_comp
    
    return row

#Make the data types of meta correct
def correct_schema_meta(df_meta_adjusted):
    df_meta_adjusted['start_time'] = pd.to_datetime(df_meta_adjusted['start_time'], utc=True)
    df_meta_adjusted['stop_time'] = pd.to_datetime(df_meta_adjusted['stop_time'], utc=True)

    df_meta_adjusted['start_time'] = (df_meta_adjusted['start_time'].astype('int64') // 10**6).astype('int64') 
    df_meta_adjusted['stop_time'] = (df_meta_adjusted['stop_time'].astype('int64') // 10**6).astype('int64')

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


    cleaned_file_path = 'traces/surfsara_trace/trace/meta.parquet'
    pq.write_table(table, cleaned_file_path)

def save_average_job_completion():
    total_completion_rate = 0
    num_items = len(job_completion)

    for value in job_completion:

        total_completion_rate += value

    average_completion_rate = total_completion_rate / num_items


#Make the datatypes of trace correct
def correct_schema_trace(df_trace_adjusted):
    df_trace_adjusted['timestamp'] = pd.to_datetime(df_trace_adjusted['timestamp'], errors='coerce')

    df_trace_adjusted['timestamp'] = (df_trace_adjusted['timestamp'].astype('int64') // 10**6).astype('int64')

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

    cleaned_file_path = 'traces/surfsara_trace/trace/trace.parquet'
    pq.write_table(table, cleaned_file_path)


def main(cpu_speed_mhz, cpu_count):
    global system_cpu_capacity_per_node, final_scaling_factor

    system_cpu_capacity_per_node = cpu_speed_mhz * cpu_count
    final_scaling_factor = system_cpu_capacity_per_node / trace_capacity_per_node
        
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
    print("returned from correcting")
    
      

    if new_rows_trace and new_rows_meta: 
        new_df_trace = pd.DataFrame(new_rows_trace)
        df_trace_adjusted = pd.concat([df_trace_adjusted, new_df_trace], ignore_index=True)

        new_df_meta = pd.DataFrame(new_rows_meta)
        df_meta_adjusted = pd.concat([df_meta_adjusted, new_df_meta], ignore_index=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: surf_workload_scaling.py <cpu_speed_mhz> <cpu_count>")
        sys.exit(1)
        
    cpu_speed_mhz = float(sys.argv[1])
    cpu_count = int(sys.argv[2])

    main(cpu_speed_mhz, cpu_count)