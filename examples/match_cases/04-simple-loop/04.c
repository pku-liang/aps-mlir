int loop(int a, int b) {
    volatile int c = 0;
    for (int i = 0; i < 10; i++) {
        c += (a * b);
    }
    return c;
}