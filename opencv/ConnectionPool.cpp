#include "ConnectionPool.h"



ConnectionPool* ConnectionPool::getConnectPool()
{
	//static ConnectionPool pool; // ��̬�ֲ����󣺵ڶ��ε���ʱ�Ѿ����ڣ��õ�����ͬһ���ڴ��ַ
	return (new ConnectionPool("root", "ljh307999", "test", "127.0.0.1", 3306));
}

ConnectionPool::ConnectionPool(string user, string passwd, string dbName, string ip, unsigned short port = 3306)
{
	m_user = user;
	m_passwd=passwd;
	m_dbName = dbName;
	m_ip = ip;
	m_port = port;

	for (int i = 0; i < m_minSize; i++)
	{
		addConnection();    // �������
	}
	thread producer(&ConnectionPool::produceConnection, this);   // ���� std::thread ������ĳ�Ա������Ҫ�������һ������ָ����Ϊ����
	thread recycler(&ConnectionPool::recycleConnection, this);   // ����
	producer.detach();
	recycler.detach();  // ���̷߳���
}

shared_ptr<MysqlConn> ConnectionPool::getConnection()
{
	unique_lock<mutex> locker(m_mutexQ);
	while (m_connectionQ.empty())
	{
		if (cv_status::timeout == m_cond.wait_for(locker, chrono::milliseconds(m_timeout)));  // waitһ��ʱ�� ms
		{
			// ˫�ؼ�飬˵�������������Ϊ��
			if (m_connectionQ.empty())  // �о�����ֱ����wait
			{
				//return nullptr;
				continue;
			}
		}
	}
	shared_ptr<MysqlConn> connptr(m_connectionQ.front(), [this](MysqlConn* conn)    // �ڶ���������ָ��ɾ���� ���ض��壩
		{
			lock_guard<mutex> locker(m_mutexQ); // ��locker������ʱ���Ƚ����Զ�������������
			//m_mutexQ.lock();
			conn->refreshAliveTime();
			m_connectionQ.push(conn);   // ����,������Դ��Ҫ����
			//m_mutexQ.unlock();
		});
	m_connectionQ.pop();
	m_cond.notify_all();    // ����������,��������û��Ӱ�죨��������ظ�ִ��һ��ѭ����
	return connptr;
}

ConnectionPool::~ConnectionPool()
{
	while (!m_connectionQ.empty())
	{
		MysqlConn* conn = m_connectionQ.front();
		m_connectionQ.pop();
		delete conn;
	}
}

void ConnectionPool::produceConnection()
{
	while (true)
	{
		unique_lock<mutex> locker(m_mutexQ);    // ��װ��һ��������������locker�����Զ�����������
		while (m_connectionQ.size() >= m_minSize)   // ��whileѭ��������if
		{
			m_cond.wait(locker);
		}
		addConnection();
		m_cond.notify_all();    // ����������,������������������ߣ��������߻������ڵڶ���whileѭ��
	}
}

void ConnectionPool::recycleConnection()
{
	while (true)
	{
		this_thread::sleep_for(chrono::seconds(1)); // ����1s
		lock_guard<mutex> locker(m_mutexQ);
		while (m_connectionQ.size() > m_minSize)    // ֻ�д�����С������Ҫ����
		{
			MysqlConn* conn = m_connectionQ.front();    // ȡ����ͷ����Ϊ��ͷʱ���
			if (conn->getAliveTime() >= m_maxIdleTime)
			{
				m_connectionQ.pop();
				delete conn;
			}
			else
			{
				break;  // ʲôҲ����
			}
		}
	}
}

void ConnectionPool::addConnection()
{
	MysqlConn* conn = new MysqlConn;
	if (!conn->connect(m_user, m_passwd, m_dbName, m_ip, m_port))
	{
		cout << "����ʧ�ܣ�" << endl;
		exit(0);
	}
	conn->refreshAliveTime();   // ˢ�¿���ʱ��
	m_connectionQ.push(conn);
}
