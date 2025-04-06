#!/usr/bin/env python3
import argparse
import replicate
import requests
from pathlib import Path
import trimesh
import tempfile
import traceback

def safe_relpath(path, base=Path.cwd()):
    try:
        return path.relative_to(base)
    except ValueError:
        return path

class Image2MeshConverter:
    """
    画像から Replicate 経由で 3D メッシュ (glb) を生成し、
    同じディレクトリに .glb および .obj ファイルとして保存するクラス。
    """
    def __init__(self, input_path: Path):
        self.input_path = input_path.resolve()
        self.glb_output = self.input_path.with_suffix(".glb")
        self.obj_output = self.input_path.with_suffix(".obj")

    def run(self):
        print(f"Running model for input: {safe_relpath(self.input_path)}")
        with open(self.input_path, "rb") as image:
            output = replicate.run(
                "ndreca/hunyuan3d-2:5d49aec561e36dc75360fad7117e8a46c7700df3070f1d24ff28509160e5f089",
                input={
                    "seed": 1234,
                    "image": image,
                    "steps": 50,
                    "guidance_scale": 5.5,
                    "octree_resolution": 512,
                    "remove_background": True,
                },
            )
        mesh_output = output.get("mesh")
        if mesh_output is None:
            print("No mesh output received from the model.")
            return
        glb_url = mesh_output.url
        print("Received GLB URL:", glb_url)
        response = requests.get(glb_url, stream=True)
        if response.status_code == 200:
            with open(self.glb_output, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"GLB file saved as: {safe_relpath(self.glb_output)}")
        else:
            print("Failed to download the glb file.")
            return
        try:
            mesh = trimesh.load(str(self.glb_output), force="mesh")
        except Exception as e:
            print(f"Error loading glb file with trimesh: {e}")
            return
        if mesh is None:
            print("Failed to load the glb file with trimesh.")
            return
        try:
            mesh.export(str(self.obj_output))
            print(f"OBJ file saved as: {safe_relpath(self.obj_output)}")
        except Exception as e:
            print(f"Error exporting obj file: {e}")

class MeshConverter:
    """
    既存の glb ファイルを読み込み、指定した形式（obj または stl）に変換・簡略化するクラス。
    --faces オプションで目標面数を指定すると、テクスチャ情報は除去してジオメトリのみを対象に、
    'meshing_decimation_quadric_edge_collapse' フィルターで簡略化を行い、
    出力ファイル名に '_faces{target_faces}' を付加します。
    """
    def __init__(self, input_glb: Path, output_type: str, target_faces: int = None):
        self.input_glb = input_glb.resolve()
        self.target_faces = target_faces
        base_stem = self.input_glb.stem
        new_stem = f"{base_stem}_faces{target_faces}" if target_faces else base_stem
        self.output_type = output_type.lower()  # "obj" または "stl"
        self.output_path = self.input_glb.with_name(new_stem + f".{self.output_type}")

    def convert(self):
        print(f"Converting mesh from: {safe_relpath(self.input_glb)}")
        # if target_faces is provided, perform decimation via pymeshlab
        if self.target_faces:
            # 1. テクスチャ情報を除去するため、まず trimesh で読み込む
            try:
                mesh = trimesh.load(str(self.input_glb), force="mesh")
                # 除去: UV やカラー情報
                if hasattr(mesh.visual, "material") and mesh.visual.material is not None:
                    mesh.visual.material.image = None
                if hasattr(mesh.visual, "uv"):
                    mesh.visual.uv = None
            except Exception as e:
                print(f"Error loading mesh with trimesh: {e}")
                return

            # 2. 一時ファイルとして、テクスチャなしの PLY 形式にエクスポート（PLY はテクスチャ情報を持たない）
            temp_glb = Path(tempfile.mktemp(suffix=".ply"))
            try:
                mesh.export(str(temp_glb))
                print(f"Temporary mesh without texture saved as: {safe_relpath(temp_glb)}")
            except Exception as e:
                print(f"Error exporting temporary mesh: {e}")
                return

            # 3. pymeshlab を用いて簡略化処理を実行
            try:
                import pymeshlab as ml
            except ImportError:
                print("pymeshlab is not installed; cannot perform decimation.")
                return

            try:
                ms = ml.MeshSet()
                ms.load_new_mesh(str(temp_glb))
            except Exception as e:
                print(f"Error loading temporary mesh into pymeshlab: {e}")
                return

            try:
                print(f"Simplifying mesh to {self.target_faces} faces (texture discarded)...")
                ms.apply_filter(
                    'meshing_decimation_quadric_edge_collapse',
                    targetfacenum=self.target_faces,
                    targetperc=0.0,         # 正確な面数目標
                    qualitythr=0.3,
                    preserveboundary=True,
                    boundaryweight=1.0,
                    optimalplacement=True,
                    preservenormal=True,
                    preservetopology=True,
                    planarquadric=True,
                    selected=False
                )
                ms.save_current_mesh(str(self.output_path))
                print(f"Mesh saved as: {safe_relpath(self.output_path)}")
            except Exception as e:
                print(f"Error during mesh simplification: {e}")
                try:
                    print("Available filters:")
                    ms.print_filter_script()
                except Exception as ex:
                    print("Error printing filter script:", ex)
            finally:
                try:
                    temp_glb.unlink()
                except Exception as e:
                    print(f"Warning: failed to remove temporary file: {e}")
        else:
            # 単純な形式変換のみの場合
            try:
                mesh = trimesh.load(str(self.input_glb), force="mesh")
            except Exception as e:
                print(f"Error loading glb file with trimesh: {e}")
                return
            if mesh is None:
                print("Failed to load the glb file with trimesh.")
                return
            try:
                mesh.export(str(self.output_path))
                print(f"Mesh saved as: {safe_relpath(self.output_path)}")
            except Exception as e:
                print(f"Error exporting mesh file: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Hunyuan3D-2 の 3D メッシュ生成・変換ツール。\n"
                    "・--input を指定すると、画像からメッシュ生成を行い、.glb および .obj ファイルとして保存します。\n"
                    "・--convert を指定すると、既存の glb ファイルを指定した形式（obj または stl）に変換・簡略化します。\n"
                    "   --faces オプションで目標面数を指定すると、出力ファイル名に '_faces{値}' を付加します。\n"
                    "※ テクスチャ情報は保持せず、ジオメトリのみを対象に処理します。"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input",
        type=Path,
        help="入力画像ファイルのパス (例: ./my_image.png)。画像からメッシュ生成を行います。",
    )
    group.add_argument(
        "--convert",
        type=Path,
        help="変換する glb ファイルのパス (例: ./my_object.glb)。指定した形式に変換・簡略化します。",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="obj",
        choices=["obj", "stl"],
        help="変換先のファイル形式。obj または stl を指定 (デフォルト: obj)。",
    )
    parser.add_argument(
        "--faces",
        type=int,
        default=None,
        help="目標面数を指定します (例: 10000)。指定された場合、簡略化処理を行い、出力ファイル名に '_faces{値}' を付加します。",
    )
    args = parser.parse_args()

    if args.input:
        converter = Image2MeshConverter(args.input)
        converter.run()
    elif args.convert:
        converter = MeshConverter(args.convert, args.type, args.faces)
        converter.convert()

if __name__ == "__main__":
    main()