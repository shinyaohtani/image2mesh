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
        # 出力 OBJ は input の stem 名のサブフォルダ内に保存
        self.subdir = self.input_path.parent / self.input_path.stem
        self.subdir.mkdir(exist_ok=True)
        self.obj_output = self.subdir / (self.input_path.stem + ".obj")

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
    --faces オプションで目標面数を指定すると、テクスチャ情報は無視してジオメトリのみを対象に、
    'meshing_decimation_quadric_edge_collapse' フィルターで簡略化を行い、
    出力ファイル名に '_faces{target_faces}' を付加します。
    また、--scale オプションで指定した倍率を適用して単位変換（例：メートル→ミリメートル）を行います。
    """

    def __init__(
        self,
        input_glb: Path,
        output_type: str,
        target_faces: int = None,
        scale: float = 1.0,
    ):
        self.input_glb = input_glb.resolve()
        self.target_faces = target_faces
        self.scale = scale
        base_stem = self.input_glb.stem
        new_stem = f"{base_stem}_faces{target_faces}" if target_faces else base_stem
        # 出力フォルダとして、新しい stem 名のサブフォルダを作成
        self.subdir = self.input_glb.parent / new_stem
        self.subdir.mkdir(exist_ok=True)
        self.output_type = output_type.lower()  # "obj" または "stl"
        self.output_path = self.subdir / (new_stem + f".{self.output_type}")

    def convert(self):
        print(f"Converting mesh from: {safe_relpath(self.input_glb)}")
        try:
            mesh = self._load_mesh()
            mesh = self._apply_scale(mesh)
            temp_glb = self._export_temp_mesh(mesh)
            simplified_mesh = self._simplify_mesh(temp_glb)
            self._export_final_mesh(simplified_mesh)
        except Exception as e:
            print(f"Error during conversion: {e}")

    def _load_mesh(self):
        try:
            mesh = trimesh.load(str(self.input_glb), force="mesh")
            if hasattr(mesh.visual, "material") and mesh.visual.material is not None:
                mesh.visual.material.image = None
            if hasattr(mesh.visual, "uv"):
                mesh.visual.uv = None
            return mesh
        except Exception as e:
            raise RuntimeError(f"Error loading mesh with trimesh: {e}")

    def _apply_scale(self, mesh):
        if self.scale != 1.0:
            try:
                print(f"Applying scale factor of {self.scale} to mesh...")
                mesh.apply_scale(self.scale)
            except Exception as e:
                raise RuntimeError(f"Error applying scale factor: {e}")
        return mesh

    def _export_temp_mesh(self, mesh):
        temp_glb = Path(tempfile.mktemp(suffix=".ply"))
        try:
            mesh.export(str(temp_glb))
            print(f"Temporary mesh without texture saved as: {safe_relpath(temp_glb)}")
            return temp_glb
        except Exception as e:
            raise RuntimeError(f"Error exporting temporary mesh: {e}")

    def _simplify_mesh(self, temp_glb):
        try:
            import pymeshlab as ml
        except ImportError:
            raise RuntimeError("pymeshlab is not installed; cannot perform decimation.")

        try:
            ms = ml.MeshSet()
            ms.load_new_mesh(str(temp_glb))
            if self.target_faces:
                print(f"Simplifying mesh to {self.target_faces} faces...")
                ms.apply_filter(
                    "meshing_decimation_quadric_edge_collapse",
                    targetfacenum=self.target_faces,
                    targetperc=0.0,
                    qualitythr=0.3,
                    preserveboundary=True,
                    boundaryweight=1.0,
                    optimalplacement=True,
                    preservenormal=True,
                    preservetopology=True,
                    planarquadric=True,
                    selected=False,
                )
            return ms
        except Exception as e:
            raise RuntimeError(f"Error during mesh simplification: {e}")
        finally:
            try:
                temp_glb.unlink()
            except Exception as e:
                print(f"Warning: failed to remove temporary file: {e}")

    def _export_final_mesh(self, ms):
        try:
            ms.save_current_mesh(str(self.output_path))
            print(f"Mesh saved as: {safe_relpath(self.output_path)}")
        except Exception as e:
            raise RuntimeError(f"Error exporting final mesh: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Hunyuan3D-2 の 3D メッシュ生成・変換ツール。\n"
        "・--input を指定すると、画像からメッシュ生成を行い、.glb および .obj ファイルとして保存します。\n"
        "・--convert を指定すると、既存の glb ファイルを指定した形式（obj または stl）に変換・簡略化します。\n"
        "   --faces オプションで目標面数を指定すると、出力ファイル名に '_faces{値}' を付加します。\n"
        "   --scale オプションでスケールファクターを指定すると、単位変換（例：メートル→ミリメートル）が行われます。\n"
        "※ glTF (glb) には明示的な単位情報は含まれていません。通常はメートル単位と仮定されるため、Fusion360 で mm として扱うには 1000 倍の変換が必要です。"
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
    parser.add_argument(
        "--scale",
        type=float,
        default=1000,
        help="スケールファクターを指定します。glb は通常メートル単位ですが、Fusion360 で mm 単位で利用する場合は 1000 を指定 (デフォルト: 1000)。",
    )
    args = parser.parse_args()

    if args.input:
        converter = Image2MeshConverter(args.input)
        converter.run()
    elif args.convert:
        converter = MeshConverter(args.convert, args.type, args.faces, args.scale)
        converter.convert()


if __name__ == "__main__":
    main()
