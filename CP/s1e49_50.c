#include <stdio.h>
#include <stdlib.h>

typedef int INTEGER;
typedef int *PINTEGER;
// typedef int INTEGER, *PINTEGER;
typedef int (*PA)[3];
typedef int (*PFUN)(void);
typedef int *(*PFUN)(int);

typedef struct Date
{
    int year;
    int month;
    int day;
} DATE, *PDATE;

int fun(void)
{
    return 520;
}

int main1(void)
{
    INTEGER a = 520;
    PINTEGER b,c;

    b = &a;
    c = b;

    printf("c = %d\n", *c);

    return 0;
}

int main2(void)
{
    struct Date *date;

    // date = (struct Date*)malloc(sizeof(struct Date));
    // date = (DATE *)malloc(sizeof(DATE));
    date = (PDATE)malloc(sizeof(DATE));

    if (date == NULL)
    {
        printf("内存分配失败!\n");
        exit(1);
    }

    date->year = 2020;
    date->month = 11;
    date->day = 11;

    printf("%d-%d-%d\n", date->year, date->month, date->day);

    return 0;
}

int main3(void)
{
    int array[3] = {1, 2, 3};
    PA pa = &array;

    for (int i = 0; i < 3; i++)
    {
        printf("%d\n",(*pa)[i]);
    }

    return 0;
}

int main(void)
{
    PFUN pfun = &fun;

    printf("%d/n", (*pfun)());

    return 0;
}