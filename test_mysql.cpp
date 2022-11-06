#include"MysqlConn.h"
#include"ConnectionPool.h"
void op1(int begin, int end)
{
	for (int i = begin; end; i++)
	{
		MysqlConn conn;
		bool judge = conn.connect("root", "ljh307999", "test", "127.0.0.1");	//MysqlĬ�ϲ��Ὺ������IP���ӵ�Ȩ�ޣ���Ҫ�ֶ�����
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
		shared_ptr<MysqlConn> conn = pool->getConnection();	// ��������ָ��
		char sql[1024] = { 0 };
		sprintf_s(sql, "insert into person values(%d,'Mary',20);", i);
		bool flag = conn->update(sql);
		cout << "flag value:" << flag << endl;
	}
}

void test()	// ���߳����ӳ�
{
	ConnectionPool* pool = ConnectionPool::getConnectPool();	// �û���Ϊ��root!

	thread t1(op2, pool, 0, 1000);
	thread t2(op2, pool, 1000, 2000);
	cout << "hello";
	t1.join();
	t2.join();

	delete pool;


	/*
	���������ʧ�ܣ���Ϊ���߳�ʹ����ͬ���û�������ȥͬʱ��¼��mysql��ܾ�һЩ����
	*/
	// ���̷߳����ӳ� 
	//MysqlConn conn;
	//bool judge = conn.connect("root", "ljh307999", "test", "127.0.0.1");	//MysqlĬ�ϲ��Ὺ������IP���ӵ�Ȩ�ޣ���Ҫ�ֶ�����
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
	bool judge = conn.connect("root", "ljh307999", "test", "127.0.0.1");	//MysqlĬ�ϲ��Ὺ������IP���ӵ�Ȩ�ޣ���Ҫ�ֶ�����
	if (!judge)
	{
		cout << "connect error";
		return -1;
	}
	string sql = "insert into person values(4,'Mary',20);";
	bool flag = conn.update(sql);
	cout << "flag value:" << flag << endl;

	sql = "select * from person;";
	conn.query(sql);	// ��ѯ���ݿ�
	while (conn.next())
	{
		cout << conn.Value(0) << " ," << conn.Value(1) << " ," << conn.Value(2) << endl;
	}
	return 0;
}