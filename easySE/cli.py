import argparse
from easySE import rssi, gfdir2gf

def main():
    parser = argparse.ArgumentParser(prog="{easySE,python3 -m mpi4py -m easySE}")

    subparsers = parser.add_subparsers(dest="subcmd")

    rssi_sp = subparsers.add_parser("rssi")
    rssi.add_args(rssi_sp)

    gf_sp = subparsers.add_parser("gfdir2gf")
    gfdir2gf.add_args(gf_sp)

    args = parser.parse_args()
    subcmd = args.subcmd

    del args.subcmd
    args = vars(args)
    
    if subcmd == "rssi":
        rssi.run(**args)
    elif subcmd == "gfdir2gf":
        gfdir2gf.run(**args)
    else:
        print(f"You either passed no subcommand or passed an invalid one!")
        parser.print_help()