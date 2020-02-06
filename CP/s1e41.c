#include <stdio.h>

#define STR(s) # s // 将s转化为一个字符串
#define TOGETHER(x, y) x ## y //记号连接运算符
#define SHOWLIST(...) printf(# __VA_ARGS__) // 可变参数
#define PRINT(format, ...) printf(# format, ## __VA_ARGS__)

inline int square(int x);

// 内联函数
inline int square(int x)
{
    return x * x;
}

int main1(void)
{
    int i = 1;

    while (i <= 100)
    {
        printf("%d的平方是%d\n", i - 1, square(i++));
    }
    return 0;
}

int main(void)
{
    printf(STR(Hello %s num = %d), STR(FishC), 520);
    printf("%d\n", TOGETHER(2,50));
    printf(SHOWLIST(nike, 520, 3.14\n));

    return 0;
}