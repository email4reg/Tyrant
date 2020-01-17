#include <stdio.h>

void getInput();

void getInput()
{
    int ch;

    if ((ch = getchar()) != '!')
    {
        getInput();
    }
    else
    {
        printf("反向输出：");
    }

    putchar(ch);
}

int main1(void)
{
    printf("请输入一句以感叹号结尾的英文句子：");
    getInput();
    putchar('\n');

    return 0;
}

unsigned int fibonacci(unsigned int n);

unsigned int fibonacci(unsigned int n)
{
    if (n == 1 || n == 2)
    {
        return 1;
    }
    else
    {
        return fibonacci(n - 1) + fibonacci(n - 2);
    }
}

int main2(void)
{
    unsigned int number, i;

    printf("请输入一个整数：");
    scanf("%u", &number);

    printf("斐波那契数列的前 %d 个数字是：", number);
    for (i = 1; i <= number; i++)
    {
        printf("%lu ", fibonacci(i));
    }
    putchar('\n');

    return 0;
}

void binary(unsigned long n);

void binary(unsigned long n)
{
    int r;

    r = n % 2;
    if (n >= 2)
    {
        binary(n / 2);
    }

    putchar('0' + r); // '0' + 1 == '1'
}

int main(void)
{
    unsigned long number;

    printf("请输入一个正整数：");
    scanf("%lu", &number);

    binary(number);
    putchar('\n');

    return 0;
}