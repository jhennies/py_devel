
import argparse

__author__ = 'jhennies'

parser = argparse.ArgumentParser(description='Write a complete name.')

parser.add_argument('Name', type=str, help='Your first name')
parser.add_argument('-sn', '--SecondName', type=str, nargs='+', help='Additional first names')
parser.add_argument('Surname', type=str, help='Your last name')

args = parser.parse_args()

outstr = args.Name
if args.SecondName is not None:
    for sn in args.SecondName :
        outstr += " " + sn
outstr += " " + args.Surname
print outstr

