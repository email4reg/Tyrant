#include <stdio.h>
#include <string.h>
#include <stdarg.h>

char *myitoa(int num, char *str);

char *myitoa(int num, char *str)
{
    int dec = 1;
    int i = 0;
    int temp;

    if (num < 0)
    {
        str[i++] = '-';
        num = -num;
    }

    temp = num;

    while (temp > 9)
    {
        dec *= 10;
        temp /= 10;
    }

    while (dec != 0)
    {
        str[i++] = num / dec + '0';
        num = num % dec;
        dec /= 10;
    }

    str[i] = '\0';

    return str;
}

int myprintf(char *format, ...);
int countInt(int num);
void printInt(int num);
void printStr(char *str);

// 这里我们使用迭代的方式打印整数
// 等后面学了递归，用递归会更方便呢
void printInt(int num)
{
    int dec = 1;
    int temp;

    if (num < 0)
    {
        putchar('-');
        num = -num;
    }

    temp = num;

    while (temp > 9)
    {
        dec *= 10;
        temp /= 10;
    }

    while (dec != 0)
    {
        putchar(num / dec + '0');
        num = num % dec;
        dec /= 10;
    }
}

// 计算整数占多少个字符
int countInt(int num)
{
    int count = 0;

    if (num < 0)
    {
        count++;
        num = -num;
    }

    do
    {
        count++;
    } while (num /= 10);

    return count;
}

void printStr(char *str)
{
    int i = 0;

    while (str[i] != '\0')
    {
        putchar(str[i]);
        i++;
    }
}

int myprintf(char *format, ...)
{
    int i = 0;
    int count = 0;
    int darg;
    char carg;
    char *sarg;
    va_list vap;

    va_start(vap, format);

    while (format[i] != '\0')
    {
        // 如果不是格式化占位符，直接打印字符串
        if (format[i] != '%')
        {
            putchar(format[i]);
            i++;
            count++;
        }
        // 如果是格式化占位符...
        else
        {
            switch (format[i + 1])
            {
            case 'c':
            {
                carg = va_arg(vap, int);
                putchar(carg);
                count++;
                break;
            }
            case 'd':
            {
                darg = va_arg(vap, int);
                printInt(darg);
                count += countInt(darg);
                break;
            }
            case 's':
            {
                sarg = va_arg(vap, char *);
                printStr(sarg);
                count += strlen(sarg);
                break;
            }
            }
            i += 2;
        }
    }

    va_end(vap);

    return count;
}

int main(void)
{
    int i;
    char str[10];

    printf("%s\n", myitoa(520, str));
    printf("%s\n", myitoa(-1234, str));


    i = myprintf("Hello %s\n", "FishC");
    myprintf("共打印了%d个字符(包含\\n)\n", i);
    i = myprintf("int: %d, char: %c\n", -520, 'H');
    myprintf("共打印了%d个字符(包含\\n)\n", i);

    return 0;
}