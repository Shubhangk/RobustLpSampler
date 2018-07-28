# hardcoded values are outputs of expermients
def Result1():
    a = [4, 0, 9, 0, 6, 0, 10, 0, 11, 0, 2, 0, 4, 0, 8, 0, 4, 0, 1, 0, 3, 0, 2, 0, 3, 0, 4, 0, 8, 0, 9, 0, 2, 0, 5, 0, 3, 0, 2, 0, 0, 0]
    b = [5, 0, 3, 0, 5, 0, 4, 0, 4, 0, 3, 0, 7, 0, 6, 0, 7, 0, 2, 0, 6, 0, 4, 0, 0, 0, 3, 0, 2, 0, 8, 0, 3, 0, 11, 0, 8, 0, 9, 0, 0, 0]
    c = [4, 0, 1, 0, 1, 0, 4, 0, 10, 0, 7, 0, 6, 0, 4, 0, 7, 0, 4, 0, 3, 0, 6, 0, 1, 0, 3, 0, 3, 0, 3, 0, 12, 0, 10, 0, 6, 0, 5, 0, 0, 0]
    d = [6, 0, 5, 0, 5, 0, 3, 0, 10, 0, 7, 0, 3, 0, 1, 0, 3, 0, 5, 0, 3, 0, 7, 0, 0, 0, 6, 0, 7, 0, 5, 0, 4, 0, 9, 0, 6, 0, 5, 0, 0, 0]
    e = [6, 0, 3, 0, 5, 0, 5, 0, 5, 0, 9, 0, 2, 0, 6, 0, 0, 0, 7, 0, 4, 0, 3, 0, 2, 0, 5, 0, 5, 0, 7, 0, 5, 0, 8, 0, 7, 0, 6, 0, 0, 0]


    p = [2, 5, 1, 2, 1, 6, 2, 3, 4, 7, 1, 6, 4, 0, 5, 0, 3, 0, 0, 3, 5, 0, 2, 5, 0, 1, 0, 1, 0, 4, 0, 4, 2, 1, 2, 4, 5, 6, 3, 0, 0, 0]
    q = [3, 2, 2, 2, 7, 2, 1, 3, 2, 7, 3, 5, 2, 2, 0, 3, 2, 0, 0, 2, 4, 5, 0, 2, 0, 1, 1, 3, 2, 2, 4, 5, 1, 1, 3, 1, 4, 7, 2, 2, 0, 0]
    r = [1, 3, 1, 3, 3, 8, 6, 1, 4, 5, 2, 5, 2, 1, 2, 2, 2, 2, 0, 1, 1, 3, 4, 1, 2, 0, 3, 1, 0, 3, 4, 3, 3, 0, 7, 1, 5, 1, 2, 2, 0, 0]
    s = [2, 1, 3, 0, 2, 2, 5, 4, 3, 3, 3, 0, 1, 1, 2, 3, 0, 3, 0, 0, 2, 5, 2, 4, 1, 0, 3, 3, 4, 4, 2, 3, 4, 2, 7, 5, 1, 6, 2, 2, 0, 0]
    t = [7, 1, 0, 2, 0, 3, 1, 2, 1, 6, 3, 6, 2, 2, 5, 0, 2, 2, 0, 1, 5, 3, 2, 4, 1, 0, 7, 0, 3, 1, 0, 1, 3, 2, 6, 5, 2, 8, 1, 0, 0, 0]

    f = [4, 0, 1, 0, 5, 0, 6, 0, 6, 0, 6, 0, 4, 0, 6, 0, 5, 0, 5, 0, 4, 0, 6, 0, 3, 0, 4, 0, 3, 0, 8, 0, 2, 0, 9, 0, 7, 0, 6, 0, 0, 0]
    u = [5, 4, 6, 2, 3, 2, 3, 4, 5, 0, 3, 0, 1, 0, 3, 2, 2, 4, 0, 1, 1, 3, 4, 0, 1, 0, 3, 3, 3, 0, 3, 3, 2, 2, 5, 5, 3, 4, 2, 3, 0, 0]
    g = [1, 0, 4, 0, 3, 0, 6, 0, 5, 0, 11, 0, 3, 0, 3, 0, 5, 0, 4, 0, 3, 0, 4, 0, 2, 0, 7, 0, 2, 0, 11, 0, 4, 0, 6, 0, 10, 0, 6, 0, 0, 0]
    v = [1, 0, 2, 2, 4, 2, 3, 2, 2, 5, 3, 5, 1, 1, 4, 0, 2, 0, 1, 0, 0, 9, 0, 3, 0, 3, 3, 1, 6, 3, 3, 4, 3, 1, 1, 8, 4, 2, 2, 4, 0, 0]


    res1 = [a[i]+b[i]+c[i]+d[i]+e[i]+f[i]+g[i] for i in range(len(a))]
    res2 = [p[i]+q[i]+r[i]+s[i]+t[i]+u[i]+v[i] for i in range(len(p))]

    print(res1)
    print(res2)

    final_lp = []
    final_rlp = []
    for i in range(len(res1)):
        if i % 2 == 1:
            continue

        final_lp.append(res1[i])
        final_rlp.append(res2[i] + res2[i+1])

    print(final_lp)
    print(final_rlp)

    import matplotlib.pyplot as plt
    x_axis = [i for i in range(len(final_lp))]
    plt.plot(x_axis, final_lp, "red", x_axis, final_rlp, "blue")
    plt.ylabel("Output Frequency")
    plt.xlabel("Sample")
    plt.show()

def Result2():
    a = [3, 0, 3, 0, 4, 0, 3, 0, 6, 0, 6, 0, 4, 0, 7, 0, 3, 0, 5, 0, 4, 0, 9, 0, 6, 0, 3, 0, 7, 0, 7, 0, 3, 0, 4, 0, 9, 0, 4, 0, 0, 0]
    p = [2, 1, 3, 3, 4, 3, 5, 2, 2, 4, 4, 3, 1, 0, 4, 0, 0, 1, 0, 4, 0, 3, 5, 2, 0, 0, 1, 1, 3, 0, 1, 5, 3, 3, 7, 4, 6, 2, 4, 4, 0, 0]

    b = [4, 0, 5, 0, 5, 0, 5, 0, 6, 0, 4, 0, 8, 0, 7, 0, 4, 0, 3, 0, 3, 0, 7, 0, 2, 0, 7, 0, 4, 0, 5, 0, 4, 0, 7, 0, 3, 0, 7, 0, 0, 0]
    q = [4, 0, 3, 0, 1, 2, 1, 4, 7, 2, 3, 3, 1, 2, 3, 0, 1, 1, 0, 0, 5, 0, 3, 5, 0, 0, 2, 5, 3, 3, 4, 2, 6, 0, 9, 3, 2, 4, 4, 2, 0, 0]

    c = [3, 0, 6, 0, 4, 0, 6, 0, 10, 0, 4, 0, 9, 0, 3, 0, 3, 0, 3, 0, 6, 0, 3, 0, 1, 0, 2, 0, 5, 0, 6, 0, 3, 0, 11, 0, 4, 0, 8, 0, 0, 0]
    r = [3, 2, 3, 3, 4, 2, 1, 3, 4, 5, 3, 5, 5, 0, 1, 9, 0, 0, 3, 0, 2, 3, 0, 3, 2, 0, 6, 0, 2, 1, 4, 4, 1, 6, 2, 2, 2, 1, 3, 0, 0, 0]

    d = [7, 0, 6, 0, 8, 0, 3, 0, 6, 0, 4, 0, 4, 0, 5, 0, 1, 0, 3, 0, 4, 0, 4, 0, 2, 0, 3, 0, 5, 0, 6, 0, 7, 0, 10, 0, 9, 0, 3, 0, 0, 0]
    s = [0, 4, 6, 2, 0, 3, 7, 3, 6, 3, 3, 6, 2, 1, 1, 1, 2, 0, 0, 0, 1, 3, 3, 3, 0, 3, 1, 2, 0, 3, 5, 2, 1, 1, 12, 2, 4, 1, 2, 1, 0, 0]

    e = [3, 0, 5, 0, 5, 0, 6, 0, 7, 0, 5, 0, 4, 0, 5, 0, 3, 0, 0, 0, 5, 0, 6, 0, 1, 0, 8, 0, 6, 0, 6, 0, 3, 0, 10, 0, 9, 0, 3, 0, 0, 0]
    t = [2, 1, 1, 4, 5, 3, 3, 5, 6, 0, 4, 3, 5, 2, 2, 2, 0, 1, 3, 0, 1, 2, 0, 3, 0, 0, 6, 5, 0, 4, 5, 3, 1, 2, 0, 4, 3, 4, 2, 3, 0, 0]

    f = [4, 0, 4, 0, 2, 0, 9, 0, 8, 0, 4, 0, 10, 0, 6, 0, 1, 0, 3, 0, 4, 0, 3, 0, 3, 0, 2, 0, 8, 0, 10, 0, 2, 0, 8, 0, 4, 0, 5, 0, 0, 0]
    u = [2, 2, 0, 3, 3, 4, 6, 2, 3, 1, 2, 5, 5, 1, 1, 2, 0, 4, 2, 3, 5, 3, 3, 1, 0, 0, 0, 2, 4, 2, 1, 2, 1, 5, 3, 2, 3, 7, 1, 4, 0, 0]

    res1 = [a[i] + b[i] + c[i] + d[i] + e[i] + f[i] for i in range(len(a))]
    res2 = [p[i] + q[i] + r[i] + s[i] + t[i] + u[i] for i in range(len(p))]

    print(res1)
    print(res2)

    final_lp = []
    final_rlp = []
    for i in range(len(res1)):
        if i % 2 == 1:
            continue

        final_lp.append(res1[i])
        final_rlp.append(res2[i] + res2[i + 1])

    print(final_lp)
    print(final_rlp)

    import matplotlib.pyplot as plt
    x_axis = [i for i in range(len(final_lp))]
    plt.plot(x_axis, final_lp, "red", x_axis, final_rlp, "blue")
    plt.ylabel("Output Frequency")
    plt.xlabel("Sample")
    plt.show()

def Result3():
    a = [4, 0, 0, 0, 1, 0, 3, 0, 5, 0, 3, 0, 2, 0, 0, 0, 1, 0, 6, 0, 0, 0, 3, 0, 1, 0, 0, 0, 3, 0, 4, 0, 4, 0, 4, 0, 1, 0, 4, 0, 0, 51]
    p = [3, 0, 0, 1, 1, 2, 2, 2, 2, 0, 7, 1, 0, 1, 1, 2, 0, 0, 0, 0, 0, 2, 1, 2, 0, 1, 2, 3, 1, 5, 2, 2, 1, 1, 2, 1, 1, 0, 2, 0, 0, 46]

    b = [1, 0, 3, 0, 4, 0, 6, 0, 6, 0, 2, 0, 3, 0, 1, 0, 2, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 3, 0, 5, 0, 4, 0, 3, 0, 1, 0, 2, 0, 0, 46]
    q = [4, 1, 3, 4, 0, 2, 0, 3, 3, 1, 0, 1, 3, 0, 0, 1, 0, 0, 0, 1, 0, 2, 2, 3, 0, 0, 0, 0, 1, 0, 2, 5, 1, 0, 1, 2, 1, 2, 1, 1, 0, 49]

    c = [2, 0, 2, 0, 5, 0, 2, 0, 10, 0, 0, 0, 4, 0, 2, 0, 2, 0, 3, 0, 0, 0, 4, 0, 1, 0, 5, 0, 3, 0, 1, 0, 0, 0, 6, 0, 6, 0, 2, 0, 0, 40]
    r = [2, 0, 0, 1, 1, 1, 1, 3, 2, 1, 4, 0, 0, 0, 0, 5, 1, 0, 0, 1, 4, 1, 3, 2, 1, 0, 0, 1, 4, 1, 0, 1, 1, 0, 2, 1, 2, 1, 4, 0, 0, 48]

    d = [2, 0, 3, 0, 1, 0, 5, 0, 7, 0, 3, 0, 2, 0, 0, 0, 1, 0, 0, 0, 4, 0, 1, 0, 0, 0, 1, 0, 6, 0, 4, 0, 3, 0, 2, 0, 4, 0, 4, 0, 0, 47]
    s = [1, 1, 3, 1, 3, 0, 0, 3, 1, 1, 2, 3, 1, 0, 1, 0, 0, 3, 0, 2, 1, 1, 0, 3, 0, 4, 2, 0, 2, 6, 4, 0, 1, 1, 3, 1, 0, 1, 0, 1, 0, 43]

    e = [4, 0, 3, 0, 3, 0, 7, 0, 2, 0, 2, 0, 3, 0, 1, 0, 3, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 3, 0, 3, 0, 3, 0, 3, 0, 0, 46]
    t = [0, 3, 2, 1, 3, 2, 2, 2, 1, 4, 0, 1, 0, 1, 0, 4, 2, 1, 1, 1, 0, 1, 3, 0, 0, 0, 3, 1, 3, 1, 4, 2, 4, 0, 1, 2, 1, 1, 1, 1, 0, 40]

    f = [5, 0, 4, 0, 3, 0, 1, 0, 2, 0, 3, 0, 3, 0, 2, 0, 1, 0, 4, 0, 3, 0, 5, 0, 2, 0, 3, 0, 4, 0, 3, 0, 4, 0, 8, 0, 5, 0, 2, 0, 0, 33]
    u = [2, 0, 0, 3, 1, 0, 1, 0, 4, 1, 1, 2, 2, 0, 0, 2, 0, 1, 0, 2, 0, 1, 2, 1, 0, 0, 2, 1, 2, 0, 2, 2, 0, 1, 1, 2, 2, 1, 0, 3, 0, 55]

    #f =
    res1 = [a[i] + b[i] + c[i] + d[i] + e[i] + f[i] for i in range(len(a))]
    res2 = [p[i] + q[i] + r[i] + s[i] + t[i] + u[i] for i in range(len(p))]

    # ONLY FOR THIS CASE OF ERROR
    res1[(len(res1) - 2)] = res1[len(res1)-1]

    print(res1)
    print(res2)

    final_lp = []
    final_rlp = []
    for i in range(len(res1)):
        if i % 2 == 1:
            continue

        final_lp.append(res1[i])
        final_rlp.append(res2[i] + res2[i + 1])

    print(final_lp)
    print(final_rlp)

    import matplotlib.pyplot as plt
    x_axis = [i for i in range(len(final_lp))]
    plt.plot(x_axis, final_lp, "red", x_axis, final_rlp, "blue")
    plt.ylabel("Output Frequency")
    plt.xlabel("Sample")
    plt.show()

Result3()
