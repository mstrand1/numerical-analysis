def bisection(a = 0, b = 0, tol = 10^-6, nmax = 300,f):
    inter = 0

    for i in range(Nmax):
        inter += 1
        p = (a+b)/2
        e = abs(b-a)

        if e < tol:
            break

        FP = (6*(exp(p)-p) - (7+3*p^2+2*p^3));

        if FP == 0
            break
        FA = 6*(exp(a)-a) - (7+3*a^2+2*a^3);
        if FA*FP > 0
            a = p
        else:
            b = p
        if i == nmax:
            i = i + 1
    if i == nmax + 1:
        print("MAX ITERATION REACHED NO SOLUTION FOUND")
    elif i < nmax + 1:
        print(p)
        print(e)
        # inter = inter - 1
        # plot(p,e,'*')
        # plottext = num2str(i);
        # dx = 0.01/(2^i);                        % Offset text
        # text(p+dx,e,cellstr(plottext))
        # title('Root Value vs. Error')
        # xlabel('Root Approximation')
        # ylabel('Error')
