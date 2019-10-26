#include <stdio.h>
#include <stdlib.h>
#include <math.h>


int func1()
{
    float m1 = 10000, m2 = 10000; // 投资额
    int count = 0; // 投资的年数

    while (m1 >= m2)
    {
        m1 = 10000 * (1 + 0.1 * count);
        m2 = 10000 * pow((1 + 0.05), count);
        count++;
    }

    printf("%d年后,黑夜的投资额超过了小甲鱼!\n", count - 1);
    printf("小甲鱼获得的本金利息和是:%.2f\n", m1);
    printf("黑夜的本金利息和为:%.2f\n", m2);

    return 0;
}

int func2()
{
    double award = 4e6;
    int count = 0;

    while (award >= 0)
    {
        award -= 5e5;
        award += award * 0.08;
        count++;
    }

    printf("%d年之后,小甲鱼败光了所有的家产,再次回到一贫如洗。。。\n", count);

    return 0;
}

int func3()
{
    int sign = 1;
    double pi = 0.0, m = 1, item = 1.0; // m表示分母, item表示前一项
    
    while (fabs(item) > 1e-8)
    {
        pi += item;
        m += 2;
        sign *= -1;
        item = sign / m;
    }

    pi = pi * 4;
    printf("Pi的近似值等于:%10.7f\n", pi); // 长度为10使得前面有个空字符，本身长度为9

    return 0;
}

int func4()
{
    long fn, f1 = 1, f2 = 1;

    for (int i = 3; i <= 24; i++)
    {
        fn = f1 + f2;
        f1 = f2;
        f2 = fn;
    }

    printf("2年后可,以繁殖%ld只兔子\n", fn);

    return 0;
}

int main()
{
    // func1();
    // func2();
    // func3();
    func4();

    return 0;
}