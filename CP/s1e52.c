#include <stdio.h>
#include <stdlib.h>
#include <time.h>



int main(void)
{
    enum Week {Sun, Mon, Tus, Wes, Thu, Fri, Sat};
    // enum Week {Sun, Mon = 10, Tus, Wes, Thu, Fri, Sat};
    enum Week today;
    struct tm *p;
    time_t t;

    time(&t);
    p = localtime(&t);

    today = p->tm_wday;

    switch (today)
    {
    case Mon:
    case Tus:
    case Wes:
    case Thu:
    case Fri:
        printf("上班!\n");
        break;
    case Sat:
    case Sun:
        printf("放假!\n");
        break;
    default:
        printf("未知错误!\n");
    }

    return 0;
}