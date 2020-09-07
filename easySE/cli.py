import argparse
from easySE import rssi, gfdir2gf

def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="subcmd")

    rssi_sp = subparsers.add_parser("rssi")
    rssi.add_args(rssi_sp)

    gf_sp = subparsers.add_parser("gfdir2gf")
    gfdir2gf.add_args(gf_sp)

    args = parser.parse_args()
    subcmd = args.subcmd
    del args.subcmd
    
    if subcmd == "rssi":
        rssi.run(**args)
    elif subcmd == "gfdir2gf":
        gfdir2gf.run(**args)
    else:
        raise ValueError(f"Shouldn't happen: What is {subcmd}?")