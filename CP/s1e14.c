#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define FIRST_KG 23
#define NEXT_KG 14

int func1()
{
    int i,j;

    for (i = 1; i <= 9; i++)
    {
        for (j = 1; j <= i; j++)
        {
            printf("%d*%d=%-2d  ", i, j, i * j);
        }
        putchar('\n');
    }

    return 0;
}

int func2()
{
    int i,price; // i为重量，price为需要支付的运费

    printf("公斤 -- 话费(元):\n");
    for (i = 1, price = FIRST_KG; i <= 20; i++, price += NEXT_KG)
    {
        printf("%3d -- %3d\n", i, price);
    }

    return 0;
}

int func3()
{
    int num = 0;
    long sum = 0L;
    int status;

    do
    {
        printf("请输入合法的数字:");
        sum += num;
        status = scanf("%d",&num);  //返回成功接收字符的个数，即1
    } while (status == 1);

    printf("结果是: %ld\n",sum);

    return 0;
}

int func4()
{
    float num = 0;
    double sum = 0;
    int status;

    do
    {
        printf("请输入合法的数字:");
        do
        {
            sum += num;
            status = scanf("%f",&num);
        } while (getchar() != '\n' && status == 1);
    } while (status == 1);

    printf("结果是: %.2lf\n", sum);

    return 0;
}

int func5() // 左上
{
    int i, j;

    for (i = 1; i <= 9; i++)
    {
        for (j = 1; j <= 9; j++)
        {
            if (i <= j)
            {
                printf("%d*%d=%-2d  ", i, j, i * j);
            }
        }
        putchar('\n');
    }

    return 0;
}


int main()
{
    // func1();
    // func2();
    // func3();
    // func4();
    func5();

    return 0;
}