import pandas as pd 

def print_metrics(metrics):
    print("\nDependency Summary:")
    for key, value in metrics.items():
        if key != 'File':
            print(f"{key}: {value}")

def write_tsv(all_metrics, output_path):
    df = pd.DataFrame(all_metrics)
    df = df.fillna(0).set_index('File')
    df = df.astype(int)
    df = df[sorted(df.columns)]
    df.to_csv(output_path, sep='\t')
