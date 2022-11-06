#include"MysqlConn.h"
#include"ConnectionPool.h"
void op1(int begin, int end)
{
	for (int i = begin; end; i++)
	{
		MysqlConn conn;
		bool judge = conn.connect("root", "ljh307999", "test", "127.0.0.1");	//Mysql默认不会开启本地IP连接的权限，需要手动开启
		if (!judge)
		{
			cout << "connect error";
			return;
		}
		char sql[1024] = { 0 };
		sprintf_s(sql, "insert into person values(%d,'Mary',20);", i);
		bool flag = conn.update(sql);
		cout << "flag value:" << flag << endl;
	}
}

void op2(ConnectionPool* pool, int begin, int end)
{
	for (int i = begin; i < end; i++)
	{
		shared_ptr<MysqlConn> conn = pool->getConnection();	// 共享智能指针
		char sql[1024] = { 0 };
		sprintf_s(sql, "insert into person values(%d,'Mary',20);", i);
		bool flag = conn->update(sql);
		cout << "flag value:" << flag << endl;
	}
}

void test()	// 多线程连接池
{
	ConnectionPool* pool = ConnectionPool::getConnectPool();	// 用户名为：root!

	thread t1(op2, pool, 0, 1000);
	thread t2(op2, pool, 1000, 2000);
	cout << "hello";
	t1.join();
	t2.join();

	delete pool;


	/*
	许多条语句会失败，因为多线程使用相同的用户和密码去同时登录，mysql会拒绝一些连接
	*/
	// 多线程非连接池 
	//MysqlConn conn;
	//bool judge = conn.connect("root", "ljh307999", "test", "127.0.0.1");	//Mysql默认不会开启本地IP连接的权限，需要手动开启
	//if (!judge)
	//{
	//	cout << "connect error";
	//	return;
	//}
	//thread t1(op1, 0, 1000);
	//thread t2(op1, 1000, 2000);
}

int query()
{
	MysqlConn conn;
	bool judge = conn.connect("root", "ljh307999", "test", "127.0.0.1");	//Mysql默认不会开启本地IP连接的权限，需要手动开启
	if (!judge)
	{
		cout << "connect error";
		return -1;
	}
	string sql = "insert into person values(4,'Mary',20);";
	bool flag = conn.update(sql);
	cout << "flag value:" << flag << endl;

	sql = "select * from person;";
	conn.query(sql);	// 查询数据库
	while (conn.next())
	{
		cout << conn.Value(0) << " ," << conn.Value(1) << " ," << conn.Value(2) << endl;
	}
	return 0;
}