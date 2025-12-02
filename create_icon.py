"""Convert PNG logo to ICO format for Windows executable."""
from PIL import Image
from pathlib import Path

def create_ico():
    assets_dir = Path(__file__).parent / "assets"
    png_path = assets_dir / "acsc_logo.png"
    ico_path = assets_dir / "acsc_logo.ico"

    if not png_path.exists():
        print(f"Error: {png_path} not found")
        return False

    img = Image.open(png_path)

    # Create multiple sizes for ICO
    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

    # Convert to RGBA if necessary
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    # Save as ICO with multiple sizes
    img.save(ico_path, format='ICO', sizes=sizes)
    print(f"Created: {ico_path}")
    return True

if __name__ == "__main__":
    create_ico()
