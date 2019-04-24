import utils

def main():
    print("HI")
    table = utils.read_table('red_wine_quality.csv')
    utils.pretty_print(table)

if __name__ == "__main__":
    main()
