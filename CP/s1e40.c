#include <stdio.h>

#define R 6371
#define PI 3.14
#define V PI * R * R * R * 4 / 3

#define MAX(x, y) ((x) > (y)) ? (x):(y)

#define SQUARE(x) ((x) * (x))

int main1(void)
{
    printf("地球的体积大概是: %.2f\n", V);

    return 0;
}

int main2(void)
{
    int x, y;

    printf("请输入两个整数:");
    scanf("%d%d", &x, &y);

    printf("%d是最大的那个数!\n", MAX(x, y));
}

int main(void)
{
    int x;

    printf("请输入一个整数: ");
    scanf("%d", &x);

    printf("%d的平方是: %d\n", x, SQUARE(x));
    printf("%d的平方是: %d\n", x+1, SQUARE(x + 1));
}
