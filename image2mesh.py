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


########################################
# Image2MeshConverter: 画像→メッシュ生成
########################################
class Image2MeshConverter:
    def __init__(self, input_path: Path):
        self.input_path = input_path.resolve()
        self.glb_output = self.input_path.with_suffix(".glb")
        self.subdir = self.input_path.parent / self.input_path.stem
        self.subdir.mkdir(exist_ok=True)
        self.obj_output = self.subdir / (self.input_path.stem + ".obj")

    def run(self):
        self._generate_glb()
        mesh = self._load_mesh(remove_texture=True)
        self._export_obj(mesh)

    def _generate_glb(self):
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
            raise RuntimeError("No mesh output received.")
        glb_url = mesh_output.url
        print("Received GLB URL:", glb_url)
        response = requests.get(glb_url, stream=True)
        if response.status_code == 200:
            with open(self.glb_output, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"GLB file saved as: {safe_relpath(self.glb_output)}")
        else:
            raise RuntimeError("Failed to download glb file.")

    def _load_mesh(self, remove_texture: bool):
        try:
            mesh = trimesh.load(str(self.glb_output), force="mesh")
            if remove_texture:
                if (
                    hasattr(mesh.visual, "material")
                    and mesh.visual.material is not None
                ):
                    mesh.visual.material.image = None
                if hasattr(mesh.visual, "uv"):
                    mesh.visual.uv = None
            return mesh
        except Exception as e:
            raise RuntimeError(f"Error loading glb with trimesh: {e}")

    def _export_obj(self, mesh):
        try:
            mesh.export(str(self.obj_output))
            print(f"OBJ file saved as: {safe_relpath(self.obj_output)}")
        except Exception as e:
            raise RuntimeError(f"Error exporting obj: {e}")


########################################
# MeshConverter: glb変換・簡略化
########################################
class MeshConverter:
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
        self.base_stem = self.input_glb.stem
        # 初期サブフォルダは入力ファイルの stem そのまま
        self.output_type = output_type.lower()  # "obj" または "stl"
        self.subdir = None  # 初期化のみ。実際の決定は _update_output_name に移動
        self.output_path = None

    def convert(self):
        print(f"Converting mesh from: {safe_relpath(self.input_glb)}")
        try:
            # --faces 未指定ならテクスチャを保持する
            remove_tex = False if self.target_faces is None else True
            mesh = self._load_mesh(remove_tex)
            mesh = self._apply_scale(mesh)
        except Exception as e:
            print(e)
            return
        if self.target_faces is None:
            self._export_direct(mesh)
        else:
            ms = self._export_and_load_temp(mesh)
            ms = self._decimate(ms)
            self._update_output_name(self.target_faces)
            self._export_final(ms)

    def _load_mesh(self, remove_texture: bool):
        try:
            mesh = trimesh.load(str(self.input_glb), force="mesh")
            if remove_texture:
                if (
                    hasattr(mesh.visual, "material")
                    and mesh.visual.material is not None
                ):
                    mesh.visual.material.image = None
                if hasattr(mesh.visual, "uv"):
                    mesh.visual.uv = None
            return mesh
        except Exception as e:
            raise RuntimeError(f"Error loading mesh: {e}")

    def _apply_scale(self, mesh):
        if self.scale != 1.0:
            try:
                print(f"Applying scale factor of {self.scale} to mesh...")
                mesh.apply_scale(self.scale)
            except Exception as e:
                raise RuntimeError(f"Error applying scale: {e}")
        return mesh

    def _export_direct(self, mesh):
        # 変換のみの場合は、直接 trimesh.export() でOBJ出力
        face_count = mesh.faces.shape[0] if hasattr(mesh, "faces") else 0
        self._update_output_name(face_count)
        try:
            mesh.export(str(self.output_path))
            print(f"Mesh saved as: {safe_relpath(self.output_path)}")
        except Exception as e:
            print(f"Error exporting mesh: {e}")

    def _export_and_load_temp(self, mesh):
        temp_path = Path(tempfile.mktemp(suffix=".ply"))
        try:
            mesh.export(str(temp_path))
            print(f"Temporary mesh saved as: {safe_relpath(temp_path)}")
        except Exception as e:
            raise RuntimeError(f"Error exporting temporary mesh: {e}")
        try:
            import pymeshlab as ml
        except ImportError:
            raise RuntimeError("pymeshlab is not installed.")
        try:
            ms = ml.MeshSet()
            ms.load_new_mesh(str(temp_path))
            return ms
        except Exception as e:
            raise RuntimeError(f"Error loading temp mesh into pymeshlab: {e}")
        finally:
            try:
                temp_path.unlink()
            except Exception:
                pass

    def _decimate(self, ms):
        try:
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
            raise RuntimeError(f"Error during decimation: {e}")

    def _get_face_count(self, ms):
        try:
            return ms.current_mesh().face_number()
        except Exception as e:
            raise RuntimeError(f"Error retrieving face count: {e}")

    def _update_output_name(self, face_count: int):
        if self.target_faces:
            new_stem = f"{self.base_stem}_faces{self.target_faces}"
            self.subdir = self.input_glb.parent / new_stem
        else:
            new_stem = f"{self.base_stem}_faces{face_count}"
            self.subdir = self.input_glb.parent / self.base_stem
        self.subdir.mkdir(exist_ok=True)
        self.output_path = self.subdir / (new_stem + f".{self.output_type}")

    def _export_final(self, ms):
        try:
            ms.save_current_mesh(str(self.output_path))
            print(f"Mesh saved as: {safe_relpath(self.output_path)}")
        except Exception as e:
            raise RuntimeError(f"Error exporting final mesh: {e}")


########################################
# main
########################################
def main():
    parser = argparse.ArgumentParser(
        description="Hunyuan3D-2 の 3D メッシュ生成・変換ツール。\n"
        "・--input: 画像からメッシュ生成。GLBはカレントフォルダに、OBJは入力ファイル名のサブフォルダに保存します。\n"
        "・--convert: 既存GLBを変換・簡略化します。\n"
        "   --faces 指定時は decimation を行い、出力フォルダ・ファイル名に '_faces{値}' を付加します。\n"
        "   --faces 未指定時は変換のみを行い、実際の面数をOBJファイル名に反映します。\n"
        "--scale: 単位変換 (例: メートル→mm、デフォルト1000)。"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input", type=Path, help="入力画像ファイルのパス (例: ./my_image.png)。"
    )
    group.add_argument(
        "--convert",
        type=Path,
        help="変換する glb ファイルのパス (例: ./my_object.glb)。",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="obj",
        choices=["obj", "stl"],
        help="変換先の形式 (obj or stl, デフォルト: obj)。",
    )
    parser.add_argument(
        "--faces",
        type=int,
        default=None,
        help="目標面数 (例: 10000)。指定時は decimation、未指定時は実際の面数を使用。",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1000,
        help="スケールファクター (例: 1000 for m→mm, デフォルト: 1000)。",
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
