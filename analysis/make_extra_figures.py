"""Reproduce the supplementary membership-function figure."""
from analysis.make_figures import load_data,fig_membership
def main():
    classical,_=load_data();fig_membership(classical)
if __name__=='__main__':main()
