# Ollama_relai configuration with MilleGrilles

## Configuration file in Coup D'Oeil

Create configuration file ollama_relai in Coup D'Oeil. In roles, put the value ollama_relai.

## Properties

### Applies to all workers

* active=1

### Task specific

These properties are organized in tasks. The keys can start with "task.1." or "task.indexing.". Each task will
spawn a number for workers (default 1) of the given type and pass the param values by name. 

Required properties.

    * task.1.type=DocumentIndexing      # Python processing module name
    * task.1.routing_keys=rk_1,rk_2     # Registers queue "a_queue" with routing keys "rk_1,rk_2".
    * task.1.active=1                   # 1 = active, anything else (including missing property) means not active.     

These are optional properties for the worker.

    * task.1.workers=1                  # Optional, number of workers to spawn per ollama_relai instance. Default 1.
    * task.1.exchange=2.prive           # Exchange for the routing keys. Default is 2.prive.
    * task.1.queue_name=a_queue         # Registers on "commande.ollama_relai.a_queue" with auto-delete. Default is anonymous exclusive queue.
    * task.1.instance_ids=e6506443-be78,instance2,...  # Comma separated list of instance_ids for the workers (filter)

These values get passed to the worker by name, e.g. DocumentIndexing(url=..., api=..., model=...). 

    * task.1.param.url=http://server.com:8000/v1
    * task.1.param.tls_method=mtls
    * task.1.param.api=openai
    * task.1.param.model=vision
    * task.1.param.context=20000
    * task.1.param.prompt=My long prompt ... for doing a specific task ...

About the tls_method: 
    
    * mtls uses the internal MilleGrille certificates including the MilleGrille's CA on both client and server during handshake.
    * external uses generally recognized CAs on the web to verify the connection (e.g. Let's Encrypt, Verisign, etc).
    * nocheck completely disabled certificate verification and just establishes and encrypted https connection.
