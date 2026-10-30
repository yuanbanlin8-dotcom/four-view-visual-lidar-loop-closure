from pathlib import Path
from zipfile import ZipFile
root=Path(__file__).parent
out=root/'manual_review'
out.mkdir(exist_ok=True)
with ZipFile(root/'manual_review_pages.zip') as z:
    for item in z.infolist():
        name=Path(item.filename)
        assert name.name==item.filename and name.suffix in ('.jpg','.csv')
        (out/name.name).write_bytes(z.read(item))
print('extracted',len(list(out.iterdir())))
