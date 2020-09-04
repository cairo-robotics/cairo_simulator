from cairo_planning.local.evaluation import SubdivisionPathIterator

if __name__ == "__main__":

    lp = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19, 20]

    spi = SubdivisionPathIterator(lp)
    for x in spi:
        print(x)

    lp2 = [[1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15], [16,17,18], [19, 20, 21]]
    ipi = SubdivisionPathIterator(lp2)

    for x in ipi:
        print(x)