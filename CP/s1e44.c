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
};

struct Book getInput(struct Book book);
// struct Book getInput(struct Book *book); // 结构体指针参数, 提高直接传入结构体的效率
void printBook(struct Book book);

struct Book getInput(struct Book book)
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

    return book;
}

void printBook(struct Book book)
{
    printf("书名: %s\n", book.title);
    printf("作者: %s\n", book.autor);
    printf("售价: %.2f\n", book.price);
    printf("出版日期: %d-%d-%d\n", book.date.year, book.date.month, book.date.day);
    printf("出版社: %s\n", book.publisher);
}

int main(void)
{
    struct Book b1, b2;
    // struct Book *b1, *b2;
    // b1 = (struct Book *)malloc(sizeof(struct Book));
    // b2 = (struct Book *)malloc(sizeof(struct Book));
    // if (b1 == NULL || b2 == NULL)
    // {
    //     printf("内存申请失败\n");
    //     exit(1);
    // }

    printf("请录入第一本书的信息...\n");
    b1 = getInput(b1);
    putchar('\n');
    printf("请录入第二本书的信息...\n");
    b2 = getInput(b2);

    printf("\n录入完毕，现在开始验证....\n");
    printf("打印第一本书的信息...\n");
    printBook(b1);
    printf("打印第二本书的信息...\n");
    printBook(b2);

    // free(b1);
    // free(b2);

    return 0;
}