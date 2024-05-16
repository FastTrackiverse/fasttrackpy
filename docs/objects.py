import sphobjinv as soi
from pathlib import Path
inv = soi.Inventory(fname_plain = Path("objects.txt"))
df = inv.data_file()
dfc = soi.compress(df)
soi.writebytes("objects.inv", dfc)