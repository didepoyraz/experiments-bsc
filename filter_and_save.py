#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import csv

def main(system_name, year, filename, filename_energy):

    df_host = pd.read_parquet(f"output/simple/raw-output/0/seed=0/host.parquet")

    #filter the hosts and sum guests running by host id 
    filtered_df = df_host[df_host['guests_running'] > 0].copy()
    
    total_energy_usage = filtered_df['energy_usage'].sum()  
    total_energy_usage_unfiltered = df_host['energy_usage'].sum()  
    total_energy_kwh = total_energy_usage / 3600000
    filtered_df['cpu_usage_mflops'] = filtered_df['cpu_usage'] * (filtered_df['cpu_time_active']/1000)
    total_cpu_mflops = filtered_df['cpu_usage_mflops'].sum()  
    total_cpu_mhz = filtered_df['cpu_usage'].sum() 
    mean_cpu_mhz = filtered_df['cpu_usage'].mean()  
    total_cpu_mhz_unfiltered = df_host['cpu_usage'].sum()  
    mean_cpu_mhz_unfiltered = df_host['cpu_usage'].mean() 
    final_metric = total_cpu_mflops / total_energy_kwh

    print("\nCalculating Pure Energy")
    print(f"Total Final Metric: {final_metric:,.2f} MFlops/KWh")
    print(f"Total Computations: {total_cpu_mflops:,.2f} MFlops")
    print(f"Mean MHZ filtered: {mean_cpu_mhz:,.2f} MHz")
    print(f"Mean MHZ Unfiltered: {mean_cpu_mhz_unfiltered:,.2f} MHz")
    print(f"Total MHZ Unfiltered: {total_cpu_mhz_unfiltered:,.2f} MHz")
    print(f"Total MHZ: {total_cpu_mhz:,.2f} MHz")
    print(f"\nFiltered Energy Used in KWh: {total_energy_kwh:,.2f} KWh")
    print(f"UNFiltered Energy Usage in Joules: {total_energy_usage_unfiltered:,.2f} Joules")
    print(f"Filtered Energy Usage in Joules: {total_energy_usage:,.2f} Joules\n\n\n\n")
    
    guests_running_per_host = filtered_df.groupby('host_id')['guests_running'].sum().reset_index()
    guests_running_per_host.columns = ['host_id', 'total_guests_running']
    
    num_active_hosts = guests_running_per_host.shape[0]
    num_all_hosts = df_host['host_id'].nunique()
    print("Number of All Hosts: ", num_all_hosts, "\nNumber of Active Hosts", num_active_hosts)

    print(f"cpu time active statistics: \n {filtered_df['cpu_time_active'].describe()}")
    data = [system_name, year, final_metric]
    data_energy =  [system_name, year, total_energy_kwh]
    
    with open(filename_energy, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_energy)
        
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

    return final_metric

if __name__ == "__main__":
    
    if len(sys.argv) != 5:
        print("Usage: filter_and_save.py <system_name> <year> <filepath> <energy_filename>")
        print(f"given input: ")
        sys.exit(1)
        
    system_name = sys.argv[1]
    year = int(sys.argv[2])
    filename = sys.argv[3]
    energy = sys.argv[4]

    main(system_name, year, filename, energy)

    