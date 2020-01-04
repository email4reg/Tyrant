#include <stdio.h>
#include <string.h>

#define MAX 1024

void func1()
{
    char str[128];

    printf("请输入鱼c工作室的域名:");
    scanf("%s", str);

    // printf("鱼c工作室的域名是: %s\n",str);
    printf("str 的地址是: %p\n", str);
    printf("str 的地址是: %p\n", &str);
    printf("str 的地址是: %p\n", &str[0]);
}

void func2()
{
    char a[] = {"FishC.com"};
    int b[] = {1, 2, 3, 4, 5};
    float c[] = {1.1, 2.1, 3.1, 4.1, 5.1};
    double d[] = {1.2, 2.2, 3.2, 4.2, 5.2};

    printf("a[0] -> %p, a[1] -> %p, a[2] -> %p, a[3] -> %p\n", &a[0], &a[1], &a[2], &a[3]);
    printf("b[0] -> %p, b[1] -> %p, b[2] -> %p, b[3] -> %p\n", &b[0], &b[1], &b[2], &b[3]);
    printf("c[0] -> %p, c[1] -> %p, c[2] -> %p, c[3] -> %p\n", &c[0], &c[1], &c[2], &c[3]);
    printf("d[0] -> %p, d[1] -> %p, d[2] -> %p, d[3] -> %p\n", &d[0], &d[1], &d[2], &d[3]);
}

void func3()
{
    char a[] = {"FishC.com"};
    int b[] = {1, 2, 3, 4, 5};
    char *pa = a;
    int *pb = b;
    // char *p = &a[0];

    printf("*p = %c, *(p+1) = %c, *(p+2) = %c\n", *pa, *(pa + 1), *(pa + 2));
    printf("*p = %d, *(p+1) = %d, *(p+2) = %d\n", *pb, *(pb + 1), *(pb + 2));
    printf("*b = %d, *(b+1) = %d, *(b+2) = %d\n", *b, *(b + 1), *(b + 2)); // 与上面相同
}

void func4()
{
    char *str = "I love FishC.com!";
    int i, length;

    length = strlen(str);

    for (i = 0; i < length; i++)
    {
        printf("%c", str[i]);
    }
    putchar('\n');
}

void func5()
{
    char str[MAX];
    char *target = str;
    int length = 0;

    printf("请输入一个字符串: ");
    fgets(str, MAX, stdin);

    while (*target++ != '\0')
    {
        length++;
    }

    printf("您总共输入了 %d 个字符！\n", length - 1);
}

void func6()
{
    char str[MAX];
    char *target = str;
    char ch;
    int length = 0;

    printf("请输入一个字符串: ");
    fgets(str, MAX, stdin);

    while (1)
    {
        ch = *target++;
        if (ch == '\0')
        {
            break;
        }
        if ((int)ch < 0)
        {
            target += 2;
        }
        length++;
    }

    printf("您总共输入了 %d 个字符！\n", length - 1);
}

void func7()
{
    char str1[MAX];
    char str2[MAX];

    char *target1 = str1;
    char *target2 = str2;

    printf("请输入一个字符串到str1中: ");
    fgets(str1, MAX, stdin);

    printf("开始拷贝str1的内容到str2中....\n");
    while ((*target2++ = *target1++) != '\0')
        ;

    printf("拷贝完毕！现在，str2中的内容是:%s\n", str2);

}


void func8()
{
    char str1[MAX];
    char str2[MAX];
    char ch;
    int n;

    char *target1 = str1;
    char *target2 = str2;

    printf("请输入一个字符串到str1中: ");
    fgets(str1, MAX, stdin);

    printf("请输入需要拷贝的字符个数:");
    scanf("%d", &n);

    printf("开始拷贝str1的内容到str2中....\n");
    while (n--)
    {
        ch = *target2++ = *target1++;
        if (ch == '\0')
        {
            break;
        }
        if ((int)ch < 0)
        {
            *target2++ = *target1++;
            *target2++ = *target1++;
        }
    }
    
    *target2 = '\0';

    printf("拷贝完毕！现在，str2中的内容是:%s\n", str2);
}


void func9()
{
    char str1[2 * MAX];
    char str2[MAX];

    char *target1 = str1;
    char *target2 = str2;

    printf("请输入第一个字符串: ");
    fgets(str1, MAX, stdin);

    printf("请输入第二个字符串: ");
    fgets(str2, MAX, stdin);

    while (*target1++ != '\0')
    {
        ;
    }
    
    target1 -= 2;

    while ((*target1++ = *target2++) != '\0')
    {
        ;
    }

    printf("连接后的结果是:%s", str1);
}

void func10()
{
    char str1[2 * MAX];
    char str2[MAX];
    int n;
    char ch;

    char *target1 = str1;
    char *target2 = str2;

    printf("请输入第一个字符串: ");
    fgets(str1, MAX, stdin);

    printf("请输入第二个字符串: ");
    fgets(str2, MAX, stdin);

    printf("请输入需要连接的字符个数:");
    scanf("%d", &n);

    while (*target1++ != '\0')
        ;
    
    target1 -= 2;

    while (n--)
    {
        ch = *target1++ = *target2++;
        if (ch == '\0')
        {
            break;
        }
        if ((int)ch < 0)
        {
            *target1++ = *target2++;
            *target1++ = *target2++;
        }
    }

    *target1 = '\0';

    printf("连接后的结果是:%s\n", str1);
}

void func11()
{
    char str1[MAX];
    char str2[MAX];

    char *target1 = str1;
    char *target2 = str2;

    int index = 1;

    printf("请输入第一个字符串: ");
    fgets(str1, MAX, stdin);

    printf("请输入第二个字符串: ");
    fgets(str2, MAX, stdin);

    while (*target1 != '\0' && *target2 != '\0')
    {
        if (*target1++ != *target2++)
        {
            break;
        }
        index++;
    }
    if (*target1 == '\0' && *target2 == '\0')
    {
        printf("两个字符串完全一致!\n");
    }
    else
    {
        printf("两个字符串不完全一致, 第%d个字符出现不同!\n", index);
    }
}

void func12()
{
    char str1[MAX];
    char str2[MAX];

    char *target1 = str1;
    char *target2 = str2;

    char ch;
    int index = 1, n;

    printf("请输入第一个字符串: ");
    fgets(str1, MAX, stdin);

    printf("请输入第二个字符串: ");
    fgets(str2, MAX, stdin);

    printf("请输入需要对比的字符个数:");
    scanf("%d", &n);

    while (n && *target1 != '\0' && *target2 != '\0')
    {
        ch = *target1;
        if (ch < 0)
        {
            if (*target1++ != *target2++ || *target1++ != *target2++)
            {
                break;
            }
        }
        if (*target1++ != *target2++)
        {
            break;
        }
        index++;
        n--;
    }
    if (*target1 == '\0' && *target2 == '\0')
    {
        printf("两个字符串完全一致!\n");
    }
    else
    {
        printf("两个字符串不完全一致, 第%d个字符出现不同!\n", index);
    }
}

int main()
{
    // func1();
    // func2();
    // func3();
    // func4();
    // func6();
    // func7();
    // func8();
    // func9();
    // func10();
    // func11();
    func12();

    return 0;
}