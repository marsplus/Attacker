package Utility;

public class Word {
	private String name;
	private String nature;
	private int feq;
	private int label;
	
	public int getLabel() {
		return label;
	}

	public int getFeq() {
		return feq;
	}
	
	public void setFeq(int feq) {
		this.feq = feq;
	}
	
	public void addFeq(){
		this.feq++;
	}
	
	public String getName() {
		return name;
	}
	
	public String getNature() {
		return nature;
	}
	
	public Word(String name, int feq) {
		this.name = name;
		this.feq = feq;
	}
	
	public Word(String name, String nature){
		this.name = name;
		this.nature = nature;
		this.feq = 1;
		this.label = -1;
	}
	
	public Word(String name, String nature, int feq, int label){
		this.name = name;
		this.nature = nature;
		this.feq = feq;
		this.label = label;
	}
	
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
	
		Word other = (Word) obj;
		return this.name.equals(other.getName());
	}
	
}
