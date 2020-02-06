#include <stdio.h>

struct Date
{
    int year;
    int month;
    int day;
};

struct Book
{
    char title[128];
    char autor[40];
    float price;
    struct Date date;
    char publisher[40];
} book = {
    "《带你学c带你飞》",
    "小甲鱼",
    48.8,
    {2017,11,11},
    "清华大学出版社"
}; // book[10]

// 创建一个名为book的Book结构体
struct Book book;
struct Book book[10]; // 结构体数组
struct Book *pt; // 结构体指针
pt = &book;
// 初始化部分成员变量
struct Book book = {.price = 48.8};
// 初始化一个结构体变量
// struct Book book = 
// {
//     "《带你学c带你飞》",
//     "小甲鱼",
//     48.8,
//     20171111,
//     "清华大学出版社"
// };

void fun1(void)
{
    struct A
    {
        char a;
        int b;
        char c;
    } a = {'x', 520, 'o'};

    printf("size of a = %d\n", sizeof(a)); // 等于12, 内存对齐！
}

int main1(void)
{
    printf("请输入书名: ");
    scanf("%s", book.title);
    printf("请输入作者: ");
    scanf("%s", book.autor);
    printf("请输入售价: ");
    scanf("%f", &book.price);
    printf("请输入出版日期: ");
    scanf("%d-%d-%d", &book.date.year, &book.date.month, &book.date.day);
    printf("请输入出版社: ");
    scanf("%s", book.publisher);

    printf("\n=======数据录入完毕=======\n");

    printf("书名: %s\n", book.title);
    printf("作者: %s\n", book.autor);
    printf("售价: %f\n", book.price);
    printf("出版日期: %d-%d-%d\n", book.date.year, book.date.month, book.date.day);
    printf("出版社: %s\n", book.publisher);

    return 0;
}

int main1(void)
{
    printf("请输入书名: ");
    scanf("%s", book.title);
    printf("请输入作者: ");
    scanf("%s", book.autor);
    printf("请输入售价: ");
    scanf("%f", &book.price);
    printf("请输入出版日期: ");
    scanf("%d%d%d", &book.date.year, &book.date.month, &book.date.day);
    printf("请输入出版社: ");
    scanf("%s", book.publisher);

    printf("\n=======数据录入完毕=======\n");

    printf("书名: %s\n", (*pt).title); // 或者pt->title
    printf("作者: %s\n", (*pt).autor);
    printf("售价: %f\n", (*pt).price);
    printf("出版日期: %d-%d-%d\n", (*pt).date.year, (*pt).date.month, (*pt).date.day);
    printf("出版社: %s\n", (*pt).publisher);

    return 0;
}
