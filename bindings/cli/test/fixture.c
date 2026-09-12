#include <arpa/inet.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static volatile sig_atomic_t running = 1;
static void terminate(int signal_number) { (void)signal_number; running = 0; }

int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "--version") == 0) {
    const char *name = strrchr(argv[0], '/');
    printf("%s " FIXTURE_VERSION "\n", name ? name + 1 : argv[0]);
    return 0;
  }
  setbuf(stdout, NULL);
  if (getenv("FIXTURE_FAIL")) { fputs("fixture startup failure\n", stderr); return 17; }
  struct sigaction action = {0};
  action.sa_handler = terminate;
  sigaction(SIGTERM, &action, NULL);
  sigaction(SIGINT, &action, NULL);
  const char *delay = getenv("FIXTURE_DELAY_MS");
  if (delay) usleep((useconds_t)atoi(delay) * 1000);
  if (!running) return 0;
  const char *address = getenv("IRONGRAPH_HTTP_ADDR");
  const char *port = strrchr(address, ':') + 1;
  int listener = socket(AF_INET, SOCK_STREAM, 0);
  int reuse = 1;
  setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  struct sockaddr_in bind_address = {0};
  bind_address.sin_family = AF_INET;
  bind_address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  bind_address.sin_port = htons((unsigned short)atoi(port));
  if (bind(listener, (struct sockaddr *)&bind_address, sizeof(bind_address)) != 0 || listen(listener, 10) != 0) {
    perror("fixture bind"); return 18;
  }
  FILE *state = fopen("fixture-persistence.txt", "a");
  fprintf(state, "start %s\n", getenv("IRONGRAPH_EXECUTION_BACKEND"));
  fclose(state);
  printf("fixture ready: %s\n", address);
  printf("fixture MCP: %s %s\n", getenv("IRONGRAPH_MCP_BINARY"), getenv("IRONGRAPH_MCP_URL"));
  while (running) {
    int client = accept(listener, NULL, NULL);
    if (client < 0) { if (errno == EINTR) continue; break; }
    char buffer[4096];
    read(client, buffer, sizeof(buffer));
    const char *response = "HTTP/1.1 200 OK\r\nContent-Length: 7\r\nConnection: close\r\n\r\nconsole";
    write(client, response, strlen(response));
    close(client);
  }
  close(listener);
  state = fopen("fixture-persistence.txt", "a");
  fputs("clean shutdown\n", state);
  fclose(state);
  puts("fixture clean shutdown");
  return 0;
}
