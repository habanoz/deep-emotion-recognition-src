import csv


def get_data(data_file):
    """Load our data from file."""
    with open(data_file, 'r') as fin:
        reader = csv.reader(fin)
        data = list(reader)

    return data


def write_data(dest_dir, data_list):
    with open(dest_dir + '/' + 'data.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_list)
