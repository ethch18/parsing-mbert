import argparse
import os
import sys
import collections

def vocabularize(filename, out_file):
    with open(filename) as input_file, \
        open(out_file, 'w') as output_file:
        ctr = collections.Counter()
        i = 1
        for line in input_file:
            if i % 100000 == 0:
                print('Read line {0} of input'.format(i))
            i += 1

            for token in line.split():
                ctr[token] += 1

        print('------Writing tokens to file------')
        output_file.write('<S>\n')
        output_file.write('</S>\n')
        output_file.write('<UNK>\n')
        
        i = 1
        for token,_ in ctr.most_common():
            if i % 1000000 == 0:
                print('Written token {0}'.format(i))
            i += 1

            output_file.write(token)
            output_file.write('\n')


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--filename', type=str, help='File to preprocess.',
        required=True)
    parser.add_argument('--output-file', type=str, help='File to output to.',
        required=True)

    args = parser.parse_args()
    
    filename = args.filename
    if not os.path.isfile(filename):
        print('ERROR: file not found')
        exit(1)

    output_file = args.output_file
    if os.path.isfile(output_file):
        
        proceed = input('Are you sure you want to override {0}? (y/n)'
            .format(output_file))
        if proceed.lower() != 'y':
            exit(0)

    vocabularize(filename, output_file)

if __name__ == '__main__':
    main()

