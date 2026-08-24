import csv
import os
import re

base_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(base_dir, '100_customers.csv')
output_file = input_file + '.tmp'

def parse_time(t_str):
    h, m = t_str.strip().split(':')
    return int(h) * 60 + int(m)

with open(input_file, encoding='utf-8') as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

customers = []
idx = 4
while idx < len(lines):
    m = re.match(r'^(\d+)\.$', lines[idx])
    if not m:
        idx += 1
        continue
    c_id = m.group(1)

    idx += 1
    if lines[idx] != "Địa chỉ:":
        continue

    idx += 1
    address = lines[idx]

    idx += 1
    if lines[idx] != "Khối lượng:":
        continue

    idx += 1
    demand = int(lines[idx].replace('kg', '').strip())

    idx += 1
    if lines[idx] != "Thời gian:":
        continue

    idx += 1
    times = lines[idx].split('-')
    ready_time = parse_time(times[0].strip())
    due_time = parse_time(times[1].strip())

    customers.append({
        'name': f'Khách hàng {c_id}',
        'address': address,
        'demand': demand,
        'ready': ready_time,
        'due': due_time,
        'service': 10
    })

    idx += 1

with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['name', 'address', 'demand', 'ready', 'due', 'service'])
    writer.writerow(['Kho - ĐH Tôn Đức Thắng', '19 Nguyễn Hữu Thọ, Phường Tân Hưng, TP.HCM', 0, 0, 1440, 0])
    for c in customers:
        writer.writerow([c['name'], c['address'], c['demand'], c['ready'], c['due'], c['service']])

os.replace(output_file, input_file)
print(f"Processed {len(customers)} customers.")
