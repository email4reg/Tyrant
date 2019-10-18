#include <stdio.h>
#include <math.h>


int func1()
{
    signed char i;
    unsigned char j;

    i = 255;
    j = 255;

    printf("i的值是%hhd\n",i);
    printf("j的值是%hhu\n",j);

    return 0;
}

int func2()
{
    int i;

    printf("请输入一个整数:");
    scanf("%d",&i);

    printf("%d的5次方是：%.2f\n", i, pow(i, 5));

    return 0;
}

int main()
{
    // func1();
    func2();

    return 0;
}