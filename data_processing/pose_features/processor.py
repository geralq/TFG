import pandas as pd
import os
import shutil
from data_processing.pose_features.pose_geometry import pose_distances
from data_processing.pose_features.utils import euclidean_distance

def process_directory(input_dir, output_dir, clean_output_dir=True):
    if os.path.exists(output_dir):
        if clean_output_dir:
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        os.makedirs(output_dir)

    print(f"Directory {output_dir} created")

    for subfolder in os.listdir(input_dir):
        subfolder_path = os.path.join(input_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        output_subfolder_path = os.path.join(output_dir, subfolder)
        os.makedirs(output_subfolder_path, exist_ok=True)

        for file in os.listdir(subfolder_path):
            if not file.endswith('.csv') or file.endswith('_features.csv'):
                continue

            file_path = os.path.join(subfolder_path, file)
            df = pd.read_csv(file_path)
            output_rows = []
            l = df.columns[1:]

            for _, row in df.iterrows():
                dict_coords = {'frame': row['frame']}
                for i in range(0, len(l), 3):
                    name = l[i].split('_x')[0]
                    dict_coords[name] = (row[l[i]], row[l[i+1]], row[l[i+2]])

                d1 = euclidean_distance(dict_coords['pose.LEFT_HIP'], dict_coords['pose.LEFT_SHOULDER'])
                d2 = euclidean_distance(dict_coords['pose.RIGHT_HIP'], dict_coords['pose.RIGHT_SHOULDER'])
                height_percentage = ((d1 + d2) / 2) * 0.265

                distances = pose_distances(dict_coords)

                for k in distances:
                    distances[k] = distances[k] / height_percentage

                frame_data = {'frame': row['frame'], **distances}
                output_rows.append(frame_data)

            df_out = pd.DataFrame(output_rows)
            output_file = os.path.join(output_subfolder_path, file.replace('.csv', '_features.csv'))
            df_out.to_csv(output_file, index=False)
            print(f"{output_file}")

def process_file(input_csv_path):
    if not input_csv_path.endswith('.csv') or input_csv_path.endswith('_features.csv'):
        print("Archivo no válido. Debe ser un archivo .csv sin '_features'.")
        return

    if not os.path.exists(input_csv_path):
        print(f"El archivo '{input_csv_path}' no existe.")
        return

    df = pd.read_csv(input_csv_path)
    output_rows = []
    l = df.columns[1:]

    for _, row in df.iterrows():
        dict_coords = {'frame': row['frame']}
        for i in range(0, len(l), 3):
            name = l[i].split('_x')[0]
            dict_coords[name] = (row[l[i]], row[l[i+1]], row[l[i+2]])

        try:
            d1 = euclidean_distance(dict_coords['pose.LEFT_HIP'], dict_coords['pose.LEFT_SHOULDER'])
            d2 = euclidean_distance(dict_coords['pose.RIGHT_HIP'], dict_coords['pose.RIGHT_SHOULDER'])
            height_percentage = ((d1 + d2) / 2) * 0.265

            distances = pose_distances(dict_coords)
            for k in distances:
                distances[k] = distances[k] / height_percentage

            frame_data = {'frame': row['frame'], **distances}
            output_rows.append(frame_data)
        except KeyError as e:
            print(f"Error en frame {row['frame']}: punto clave faltante ({e})")
            continue

    df_out = pd.DataFrame(output_rows)

    output_file = input_csv_path.replace('.csv', '_features.csv')
    df_out.to_csv(output_file, index=False)
    print(f"Archivo generado: {output_file}")
