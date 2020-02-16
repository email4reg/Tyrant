#include <stdio.h>

int main1(void)
{
    struct Test
    {
        unsigned int a:1;
        unsigned int b:1;
        unsigned int c:2;
    } test;

    test.a = 0;
    test.b = 1;
    test.c = 2;

    printf("a = %d, b = %d, c = %d\n", test.a, test.b, test.c);
    printf("size of test = %lu\n", sizeof(test));

    return 0;
}

// 未定义行为指,在不同环境下的结果不同的一些行为,需要避免发生。
int main(void)
{
    int value = 1;

    while (value < 1024)
    {
        value <<= 1; // 左移相当于是乘以2(>0)
        printf("value = %d\n", value);
    }

    value = 1024;
    while (value > 0)
    {
        value >>= 2; // 左移相当于是除以2(>0)
        printf("value = %d\n", value);
    }
    

    return 0;
}