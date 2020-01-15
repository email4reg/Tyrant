#include <stdio.h>
#include <stdlib.h>

#define EPSINON 0.000001 // 定义允许的误差

char *getword(char);
int square(int);
int calc(int (*)(int, int), int, int);
int (*select(char))(int, int);
void *func(int n, int *ptr, char *str);

void *func(int n, int *ptr, char *str)
{
    if (n > 0)
    {
        return ptr;
    }
    else
    {
        return str;
    }
}

// 指针函数
char *getword(char ch)
{
    switch (ch)
    {
    case 'a': return "Apple";
    case 'b': return "banana";
    case 'c': return "cat";
    case 'd': return "dog";
    default: return "None";
    }
}

// 函数指针
int square(int num)
{
    return num * num;
}

// 函数指针作为参数
int add(int num1, int num2)
{
    return num1 + num2;
}

int sub(int num1, int num2)
{
    return num1 - num2;
}

float mul(float num1, float num2)
{
    return num1 * num2;
}

float divi(float num1, float num2)
{
    if (num1)
        return num1 / num2;
    else
    {
        printf("除数不能为0\n");
    }
}


int calc(int (*pf)(int, int), int num1, int num2)
{
    return (*pf)(num1, num2);
}

// 将函数指针作为返回值
int (*select(char ch))(int, int)
{
    switch(ch)
    {
        case '-': return sub;
        case '+': return add; // 函数名与数组名类似, 代表指向函数的指针，即函数指针
        default: return 0;
    }
}

int main1()
{
    char ch;
    
    printf("请输入一个字母: ");
    scanf("%c", &ch);

    printf("%s\n", getword(ch));

    return 0;
}

int main2()
{
    int num;
    int (*pf)(int);

    printf("请输入一个整型数字: ");
    scanf("%d", &num);

    pf = square; // or &square

    printf("%d * %d = %d\n", num, num, (*pf)(num));

    return 0;
}

int main3()
{
    int num1, num2;
    char ch;

    printf("请输入一个算术表达式(如:1+3): ");
    scanf("%d%c%d", &num1, &ch, &num2);

    switch(ch)
    {
        case '-': printf("%d - %d = %d\n", num1, num2, calc(sub,num1,num2));break;  // note: &sub
        case '+': printf("%d + %d = %d\n", num1, num2, calc(add,num1,num2));break;  // note: &add
    }

    return 0;
}

int main4()
{
    int num1, num2;
    char ch;
    int (*p)(int, int);

    printf("请输入一个算术表达式(如:1+3): ");
    scanf("%d%c%d", &num1, &ch, &num2);

    p = select(ch);
    printf("%d %c %d = %d\n", num1, ch, num2, calc(p, num1, num2));

    return 0;
}

int main5()
{
    int num = 520;
    char *str = "FishC";

    printf("%d\n", *(int *)(func(1, &num, str)));
    printf("%s\n", (char *)func(-1, &num, str)); // or without (char *)

    return 0;
}

// question 2
double add(double x, double y);
double sub(double x, double y);
double mul(double x, double y);
double divi(double x, double y);

double add(double x, double y)
{
    return x + y;
}

double sub(double x, double y)
{
    return x - y;
}

double mul(double x, double y)
{
    return x * y;
}

double divi(double x, double y)
{
    // 不要对浮点数进行==或!=比较，因为IEEE浮点数是一个近似值
    if (y >= -EPSINON && y <= EPSINON)
    {
        printf("除数不能为0\n");
        // 如果除数为0，调用exit()函数直接退出程序
        exit(1);
    }
    else
    {
        return x / y;
    }
}

int main()
{
    int i;
    double x, y, result;
    double (*func_table[4])(double, double) = {add, sub, mul, divi};

    printf("请输入两个数：");
    scanf("%lf %lf", &x, &y);

    printf("对这两个数进行加减乘除后的结果是：");
    for (i = 0; i < 4; i++)
    {
        result = (*func_table[i])(x, y);
        printf("%.2f ", result);
    }
    putchar('\n');

    return 0;
}