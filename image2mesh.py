#!/usr/bin/env python3
import argparse
import replicate
import requests
from pathlib import Path
import trimesh

class Image2MeshConverter:
    def __init__(self, input_path: Path):
        # 入力画像の絶対パスを取得
        self.input_path = input_path.resolve()
        # 入力ファイルと同じベース名で、拡張子を glb と obj に変更した出力パスを設定
        self.glb_output = self.input_path.with_suffix('.glb')
        self.obj_output = self.input_path.with_suffix('.obj')

    def run(self):
        # Replicate のモデルを実行する
        # ローカルファイルの場合、replicate ライブラリはファイルパス（または文字列）を受け付けます
        print(f"Running model for input: {self.input_path}")
        output = replicate.run(
            "ndreca/hunyuan3d-2:5d49aec561e36dc75360fad7117e8a46c7700df3070f1d24ff28509160e5f089",
            input={
                "seed": 1234,
                "image": str(self.input_path),
                "steps": 20,
                "guidance_scale": 5.5,
                "octree_resolution": 512,
                "remove_background": True
            }
        )
        # 出力は辞書形式で、"mesh" キーに FileOutput オブジェクトが入っています
        mesh_output = output['mesh']
        glb_url = mesh_output.url
        print("Received GLB URL:", glb_url)

        # URL から glb ファイルをダウンロード
        response = requests.get(glb_url, stream=True)
        if response.status_code == 200:
            with open(self.glb_output, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"GLB file saved as: {self.glb_output}")
        else:
            print("Failed to download the glb file.")
            return

        # trimesh を使って glb ファイルを読み込み、obj 形式に変換して保存
        try:
            mesh = trimesh.load(str(self.glb_output), force='mesh')
        except Exception as e:
            print(f"Error loading glb file with trimesh: {e}")
            return

        if mesh is None:
            print("Failed to load the glb file with trimesh.")
            return

        try:
            mesh.export(str(self.obj_output))
            print(f"OBJ file saved as: {self.obj_output}")
        except Exception as e:
            print(f"Error exporting obj file: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert an input image to a 3D mesh using Hunyuan3D-2 via Replicate"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input image file (e.g., ./my_image.png)"
    )
    args = parser.parse_args()

    converter = Image2MeshConverter(args.input)
    converter.run()

if __name__ == "__main__":
    main()
