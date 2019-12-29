#include <stdio.h>
#include <stdbool.h>

void func1()
{
    char a = 'F';
    int f = 123;

    char *pa = &a;
    int *pb = &f;

    printf("a = %c\n", *pa);
    printf("f = %d\n", *pb);

    *pa = 'C';
    *pb += 1;

    printf("a = %c\n", *pa);
    printf("f = %d\n", *pb);

    printf("sizeof pa = %lu\n", sizeof(pa));
    printf("sizeof pb = %lu\n", sizeof(pb));

    printf("the address of a is %p\n", pa);
    printf("the address of f is %p\n", pb);

}

void func2()
{
    int a;
    int *pa = &a;

    printf("请输入一个整数:");
    scanf("%d",&a);

    printf("a = %d\n", a);

    printf("请重新输入一个整数:");
    scanf("%d",pa);

    printf("a = %d\n", *pa);

}


void fun3()
{
    int a, b, c, t;
    int *pa, *pb, *pc;

    printf("请输入三个数:");
    scanf("%d %d %d", &a, &b, &c);

    pa = &a;
    pb = &b;
    pc = &c;

    if (a > b)
    {
        t = *pa;
        *pa = *pb;
        *pb = t;
    }
    
    if (a > c)
    {
        t = *pa;
        *pa = *pc;
        *pc = t;
    }

    if (b > c)
    {
        t = *pb;
        *pb = *pc;
        *pc = t;
    }

    printf("%d <= %d <= %d\n", *pa, *pb, *pc);
    // printf("%d <= %d <= %d\n", a, b, c);

}

int func4()
{
    int i, j, n, cubed, sum = 0;

    printf("请输入一个整数:");
    scanf("%d",&n);

    cubed = n * n * n;

    for (i = 1; i < cubed; i += 2)
    {
        for (j = i; j < cubed; j += 2)
        {
            sum += j;
            if (sum == cubed)
            {
                if (j - i > 4)
                {
                    printf("%d = %d + %d ... + %d\n", cubed, i, i + 2, j);
                }
                else
                {
                    printf("%d = %d + %d + %d\n", cubed, i, i + 2, i + 4);
                }
                goto FINDIT;
                
            }

            if (sum > cubed)
            {
                sum = 0;
                break;
            }
        }
    }

FINDIT:
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
