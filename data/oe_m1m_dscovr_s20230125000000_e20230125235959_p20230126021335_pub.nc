CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230125000000_e20230125235959_p20230126021335_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-01-26T02:13:35.103Z   date_calibration_data_updated         2022-11-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-01-25T00:00:00.000Z   time_coverage_end         2023-01-25T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        e   records_fill         ;   records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx^c��  "          @�z��9��@�Q�@��
BffB�k��9��@��@�\)B`p�C�                                    Bx^c�&  
�          @�p��,��@��@^�RA�ffB����,��@K�@�BL��Cc�                                    Bx^c��  T          @����
@�(�@EA�Q�B�z���
@l��@�\)BC{B���                                    Bx^c�r  T          @�zῡG�@�p�@��
B(G�B�G���G�@"�\@��
B�ffB��                                    Bx^c�  T          @�\)��=q@I��@vffB��CB���=q?�{@���B@\)C!�{                                    Bx^c߾  �          @����<��@c33@�z�B.
=C���<��?�G�@�{Bj�C�                                    Bx^c�d  �          @����8��@�{@q�B�B�.�8��@$z�@��BV33C
n                                    Bx^c�
  
�          @�{����@�z�@�  B�RBߏ\����@*=q@���Bn=qB��q                                    Bx^d�  �          @�
=�w�@���@(Q�A�33C� �w�@*=q@���B"G�C}q                                    Bx^dV  T          @˅���\?�
=���R�YC!�����\@��
=��z�C�f                                    Bx^d(�  �          @�33����?�33����b�\C!������@
=�(����
=C�)                                    Bx^d7�  T          @ʏ\���
?p�׿�p��4  C+G����
?��ͿB�\��(�C'��                                    Bx^dFH  	�          @��
���R?�
=���R�Y�C$@ ���R@�ÿ333�ʏ\C G�                                    Bx^dT�  
�          @�33�Ǯ?�
=�E����C$���Ǯ?��ͽ#�
��{C#z�                                    Bx^dc�  "          @�����?�{>��?�=qC#������?��?aG�@�33C%�
                                    Bx^dr:  �          @�z���z�?У�?(��@�\)C%�3��z�?�p�?�  A.�\C)
                                    Bx^d��  V          @����=q?�
=?G�@�
=C#  ��=q?��H?�p�AM�C&��                                    Bx^d��  �          @����
=@�>L��?�(�C!#���
=?�\)?��\A�C#@                                     Bx^d�,  �          @�  ��
=?�p�?#�
@�C%���
=?�=q?��\A.{C(^�                                    Bx^d��  
�          @׮�Ϯ?޸R��z���RC%��Ϯ?��H>�
=@eC%@                                     Bx^d�x  V          @�{���H?\(�>u@�\C,�=���H?333?\)@��\C-�3                                    Bx^d�  
�          @�{����?O\)?У�Ab�HC,�����=���?���A}C3)                                    Bx^d��  �          @�\)���?���@�\A�G�C'Y����>�Q�@)��A�C0��                                    Bx^d�j  "          @����Q�?�ff@:=qA�ffC$�f��Q�>�=q@R�\A��
C1J=                                    Bx^d�  �          @�(�����@�\@k�B33C�����>��@�{B33C/T{                                    Bx^e�  �          @����?�@3�
A�ffC"}q���?��@S33A�=qC.��                                    Bx^e\  T          @Ӆ���
@�\@)��A�p�C \���
?Q�@O\)A�C+��                                    Bx^e"  R          @�
=����?��@\)A��\C.Q����׾�{@�\A��RC70�                                    Bx^e0�  
�          @�=q��z�?��@�A�{C)xR��z�>�  @z�A�33C1�3                                    Bx^e?N  "          @��H�˅?�(�?�ffA33C)+��˅?333?�Q�AK�C-��                                    Bx^eM�  �          @��H��{?���?
=@��
C)k���{?^�R?�G�A�C,G�                                    Bx^e\�  �          @�����@Q�5��Q�Ck�����@{>�z�@!G�C��                                    Bx^ek@  �          @�(���{@ff?\A]C����{?Ǯ@z�A��
C$�f                                    Bx^ey�  �          @�\)����?n{@��B"\)C(�3���׿G�@�{B#��C=W
                                    Bx^e��  "          @���j�H?h��@�33BU�HC&!H�j�H��
=@���BR�CE�                                    Bx^e�2  
�          @���33?�@y��B��C"ٚ��33�k�@�z�B'�
C6ٚ                                    Bx^e��  T          @�  �l��@�\@�p�B5C@ �l��>�p�@�
=BT��C.T{                                    Bx^e�~  T          @��vff@#�
@��B%{CY��vff?B�\@�\)BI�C(�{                                    Bx^e�$  
�          @�ff��33@J=q@,(�A�C���33?���@mp�BQ�C�                                    Bx^e��  "          @���ff@L(�?�
=A��
Cٚ��ff@�@AG�A�(�C(�                                    Bx^e�p  "          @��H�j=q��@��BPz�C;33�j=q��@�B0\)CS�q                                    Bx^e�  "          @ƸR�O\)�W
=@�{Bd�\CB��O\)�3�
@�{B8
=C\޸                                    Bx^e��  "          @�  ����@	��@XQ�B�C&f����?(��@|��B G�C+�3                                    Bx^fb  �          @�p���z�@]p�@*=qA��C=q��z�@	��@s33B�C(�                                    Bx^f  
�          @���p�@qG�@�\A�G�C\��p�@$z�@e�B�RC!H                                    Bx^f)�  �          @�
=��33@��?��RA�  C���33@Z�H@g�B�\C(�                                    Bx^f8T  "          @�Q��]p�@�?�A��B�p��]p�@���@n{BC}q                                    Bx^fF�             @Ϯ�L(�@��?�
=AMp�B�  �L(�@�p�@[�B B��{                                    Bx^fU�  V          @�(��`  @�  ?��
A��B����`  @z=q@eB	z�C�{                                    Bx^fdF  "          @��
�_\)@�ff?��A��HB�u��_\)@tz�@j�HB�Cz�                                    Bx^fr�  �          @�{����?�@�33BA�C,�{�������@�{B8��CE�H                                    Bx^f��  "          @ƸR���?�G�@��B9p�C&Q�����Q�@���B;�\C?)                                    Bx^f�8  	�          @�
=�e>�  @��BX��C/�3�e���H@�ffBG�CM}q                                    Bx^f��  �          @�ff�B�\��R@�(�Bk\)C?�=�B�\�$z�@�\)BA�C\0�                                    Bx^f��  �          @��
����  @�  B8{CwT{����{@%AÙ�C|�f                                    Bx^f�*  �          @�{�]p���z�@�Q�BR  CF���]p��:�H@{�B#��C\�                                    Bx^f��  �          @������@_\)@��A���C{���@
=@XQ�B
=C��                                    Bx^f�v  V          @θR��{@�Q�>#�
?��RC�)��{@��H?�33A��C	c�                                    Bx^f�  T          @�=q��@aG�@ffA���C���@@a�B  Cp�                                    Bx^f��  
�          @�G����@W�@1G�A�{C����@�\@w
=B=qC��                                    Bx^gh  T          @��H��33@c33@QG�A���C)��33@ ��@�z�B/�\C��                                    Bx^g  �          @��H���H@N�R@P��A�Q�C\)���H?�(�@�  B)G�Cff                                    Bx^g"�  T          @�=q���@333@e�B
ffCE���?���@�(�B0\)C$�{                                    Bx^g1Z  �          @�����Q�@���?���A.ffC�f��Q�@QG�@*�HAƣ�C��                                    Bx^g@   �          @У���@�\)>��@fffCJ=��@~{@ffA�33C��                                    Bx^gN�  T          @�G���z�@���?�@�  C����z�@�@
=A�=qCk�                                    Bx^g]L  �          @�G��g�@����aG���G�B���g�@�\)?�p�A/\)B�aH                                    Bx^gk�  
�          @љ�����@���#�
����CL�����@�  ?�=qA��C�                                     Bx^gz�  "          @������@��
��ff��  C� ���@�p�?��HAR=qC��                                    Bx^g�>  �          @�z����
@P��@��A���C33���
@�@N�RA�p�C��                                    Bx^g��  �          @�  ��33?��R@n{Bz�C%����33��\)@z=qBQ�C7Y�                                    Bx^g��  
�          @�����(�>\@���B'(�C/J=��(����R@~�RB�
CC                                      Bx^g�0  V          @�
=����@#33@>{A��
C������?�(�@n{B�C%�3                                    Bx^g��  �          @�\)���@@c�
B=qC�f���?J=q@�{B(p�C*&f                                    Bx^g�|  �          @Ǯ��p�@ ��@u�B  C�\��p�>Ǯ@��B0�C.�                                    Bx^g�"  �          @Ǯ��p�@\)@qG�B(�C!H��p�?!G�@��HB0�
C+�                                    Bx^g��  "          @θR���R@���?�  AffC����R@e@#�
A��HC+�                                    Bx^g�n  "          @�����33@��þ�33�FffC.��33@�G�?���Ac33C��                                    Bx^h  T          @����}p�@���?��
A�RC B��}p�@�  @7�A�=qC�q                                    Bx^h�  	�          @����w�@�ff?z�@��\B�B��w�@��H@ ��A�C)                                    Bx^h*`  
�          @љ��\��@�  >L��?��
B���\��@�Q�@�A�ffB�                                      Bx^h9  �          @�  �hQ�@��R?�{AA�B����hQ�@��@O\)A�33C)                                    Bx^hG�  �          @У��W�@�33?���A_�
B�ff�W�@�33@_\)B�B��                                     Bx^hVR  �          @�  �U@�Q�?�G�A|��B����U@�ff@hQ�BG�C �                                     Bx^hd�  "          @�
=�e@���?�p�AT(�B��e@��R@Tz�A��Cz�                                    Bx^hs�  �          @�ff�j=q@�?��A�
B�k��j=q@�(�@>{A��
C�{                                    Bx^h�D  �          @�\)�U�@��?#�
@�p�B�u��U�@��H@*�HA�\)B�                                    Bx^h��  �          @θR�AG�@��>\@Z=qB�G��AG�@��H@\)A�z�B�k�                                    Bx^h��  "          @�z��7
=@�z��\���B����7
=@���?�G�Ax  B�q                                    Bx^h�6  T          @�ff�z�@ʏ\��p��i��B�L��z�@�Q�?G�@�Q�B�Q�                                    Bx^h��  �          @�\)�@���z��\��B�\�@��H?aG�@�G�B�G�                                    Bx^h˂  T          @�  � ��@�
=��p��d��B֊=� ��@���?Tz�@��HBծ                                    Bx^h�(  T          @�Q���
@�(������6{B�
=���
@�p�?��HA�B��f                                    Bx^h��  �          @�33�У�@�ff��
=�[�B�p��У�@��H?s33@�\)B��f                                    Bx^h�t  
Z          @��Ϳ�Q�@�{�����pz�B�p���Q�@�z�?J=q@��
Bϣ�                                    Bx^i  T          @����33@��
���yp�B�  ��33@ۅ?333@��
B��                                    Bx^i�  �          @�����@�(����x��B�Ǯ���@ۅ?333@�(�Bҽq                                    Bx^i#f  �          @���У�@�p���\��{B�ff�У�@�ff?(�@��
B�W
                                    Bx^i2  �          @��Ϳ�z�@ָR�G���33B�Ǯ��z�@�
=?!G�@��HB��f                                    Bx^i@�  S          @��Ϳ��@�Q��Q��|  Bɨ����@߮?8Q�@�G�B��                                    Bx^iOX  "          @�p���z�@أ׿����p(�Bˀ ��z�@�\)?L��@�{B��
                                    Bx^i]�  �          @����@�p��޸R�aB�#׿�@��H?\(�@�{B�aH                                    Bx^il�  T          @����
@أ׿˅�LQ�Bъ=���
@��
?��A�B��                                    Bx^i{J  
�          @�ff��ff@�zῳ33�3\)B�\)��ff@�p�?�G�A!G�B�B�                                    Bx^i��  �          @���ff@��H��z����Bр ��ff@���?��HA<z�Bѽq                                    Bx^i��  �          @�R�Ǯ@��
��p��>{B͙��Ǯ@�?�A��B�aH                                    Bx^i�<  T          @�
=��Q�@�Q�����M��B�  ��Q�@��
?�G�A�BӀ                                     Bx^i��  
�          @�
=���R@�\)�s33��B�����R@��H?ٙ�AYB̙�                                    Bx^iĈ  
�          @�\)����@�G��J=q��  B�B�����@ڏ\?�\)Ao�B��                                    Bx^i�.  �          @�\)��{@�=q�����K�B��Ϳ�{@�
=@\)A��\B��f                                    Bx^i��  �          @����R@߮��\��=qB�  ���R@�{@ffA��\B�\                                    Bx^i�z  T          @�ff���H@�=q�aG���\B�zῚ�H@���@��A�  BȮ                                    Bx^i�   �          @�  ����@�G��\(��ٙ�B�8R����@��
?��AeG�B���                                    Bx^j�  
�          @���z�@߮�8Q���  BΣ׿�z�@أ�?��As\)B�z�                                    Bx^jl  
�          @���  @ۅ��G��@��BД{��  @�?�{A��B�G�                                    Bx^j+  �          @����\@���\��(�B�  ��\@޸R?��@��\BԳ3                                    Bx^j9�  "          @�\)���@أ׿(���=qB����@���?�z�Aup�B�#�                                    Bx^jH^  �          @�Q��@׮��=q�)�B�L��@�Q�?�(�A33B�.                                    Bx^jW  �          @����	��@�
=��G��`Q�Bׅ�	��@���?O\)@���B֣�                                    Bx^je�  
�          @�׿�(�@�������B�  ��(�@�
=>�ff@c�
BӔ{                                    Bx^jtP  �          @��
=@�(��ٙ��YB�.�
=@�G�?Tz�@љ�B�G�                                    Bx^j��  T          @����'
=@љ��޸R�]B߀ �'
=@�\)?B�\@�  B�k�                                    Bx^j��  �          @��ÿ��R@�33�Y����\)B��)���R@�?��Ac�
B�\)                                    Bx^j�B  S          @�׿�ff@�\)�z�H��\)B��f��ff@ۅ?�\)AM��B�aH                                    Bx^j��  U          @��ÿ�{@�
=�}p���33B��ÿ�{@�33?���AJ�\B�k�                                    Bx^j��  T          @�  ��Q�@�녿Y����ffB����Q�@�z�?�G�A`��Bˮ                                    Bx^j�4  �          @�׿��
@ᙚ�\(��ڏ\B̔{���
@�(�?޸RA]��B�#�                                    Bx^j��  T          @�녿�z�@�G���=q�'�Bʔ{��z�@�G�?�ffA$Q�Bʔ{                                    Bx^j�  T          @�Q��33@��Ϳfff��p�B�
=�33@�Q�?��AQ�Bը�                                    Bx^j�&  	�          @�Q��{@޸R�8Q����RB��H��{@�Q�?�=qAjffB�                                    Bx^k�  S          @陚���R@ᙚ��33���B��)���R@߮?��HA8z�B�\                                    Bx^kr  U          @�녿��H@�
=�z�H��
=B�uÿ��H@ۅ?�=qAH(�B��                                    Bx^k$  	�          @�\��@�33�\(���
=B���@�ff?�33AP��B���                                    Bx^k2�  T          @�=q��R@�z�:�H��G�B��
��R@�ff?��
Ab{B���                                    Bx^kAd  
�          @���{@�{��\��  B�k��{@��@   A~�\B���                                    Bx^kP
  
�          @�=q�z�@�\)�=p����B��z�@�G�?�ffAc\)B��f                                    Bx^k^�  �          @�����@���R��(�B�B����@�ff?��Ao�
B�ff                                    Bx^kmV  
(          @�33��@�ff�z�H���RB����@�33?�ffAC\)B�k�                                    Bx^k{�  T          @����@�\)���R��RB�aH�@�
=?���A$Q�B�p�                                    Bx^k��  "          @�z��
=@�
=���H��\Bծ�
=@�ff?��A'�B���                                    Bx^k�H  T          @�z��\)@ᙚ��
=��BѨ���\)@�Q�?��A-�B���                                    Bx^k��  �          @��Ϳ�=q@��^�R�أ�B��Ϳ�=q@޸R?ٙ�ATz�B�ff                                    Bx^k��  
�          @��Ϳ�(�@��}p����RB����(�@��?���AH��B�u�                                    Bx^k�:  "          @�z��Q�@�33����\)BθR��Q�@���?��RA:{B���                                    Bx^k��  
�          @�z��@�����B�=q��@ۅ?�(�Aw33B�L�                                    Bx^k�  �          @�����@��
�.{����BѮ���@��?�{Aip�Bҏ\                                    Bx^k�,  T          @����
=@�\�W
=��{B�(��
=@�ff@�\A�ffB��                                    Bx^k��  �          @��H���@�녿�
=�z�B�����@���?�{A+33B�=q                                    Bx^lx  �          @�=q� ��@�Q�B�\���HB�
=� ��@�(�@�A��B�                                    Bx^l  
�          @���R@�  <#�
=���B�L���R@љ�@p�A�=qBي=                                    Bx^l+�  "          @�33�	��@�{?z�H@�
=Bր �	��@��@UA���Bڊ=                                    Bx^l:j  "          @�z���R@���=�G�?Q�B�(���R@��@#33A��Bي=                                    Bx^lI  	�          @��H�@�Q�>�  ?�p�B�(��@Ϯ@*�HA�\)Bר�                                    Bx^lW�  
�          @�33�@���>��
@!�B���@�\)@/\)A��B�                                    Bx^lf\  �          @�33�Q�@�{=#�
>���B��f�Q�@Ϯ@(�A�\)B�G�                                    Bx^lu  �          @�=q�p�@�(�=��
?(�B�k��p�@�p�@��A���B���                                    Bx^l��  �          @����0��@�\)=��
?(�B����0��@�G�@��A��B�ff                                    Bx^l�N  �          @����@�33�#�
����Bٽq�@�@ffA���B���                                    Bx^l��  �          @��/\)@��
���
�%B��H�/\)@ʏ\?�p�A�
B�q                                    Bx^l��  "          @����3�
@љ��\�B�\B�z��3�
@���?��AvffB�=q                                    Bx^l�@  �          @��<��@�G���=q�.{B�Q��<��@˅?fff@�\B���                                    Bx^l��  "          @��l(�@��\�h����\)B����l(�@��ÿ�\�g
=B��                                    Bx^lی  T          @���dz�@���aG���{B��dz�@��\?��ArffB�(�                                    Bx^l�2  �          @�\)�   @�
=����n{B��   @�\)?�{Ao�B�.                                    Bx^l��  
�          @�����
@�G�@%A��
C�3���
@g�@��B
C��                                    Bx^m~  "          @����@��\@0��A��HC�����@u@�=qB=qC	��                                    Bx^m$  �          @�
=�}p�@��
@
=qA�G�B�aH�}p�@���@|(�Bp�C�                                    Bx^m$�  
�          @�  �o\)@�?�ffAf�\B�=�o\)@�
=@mp�A�Q�B�                                      Bx^m3p  "          @�
=�o\)@�p�?�  Aa�B����o\)@�\)@i��A�G�B��H                                    Bx^mB  
�          @�ff�c33@�p�?u@�ffB��)�c33@�\)@?\)AĸRB��f                                    Bx^mP�  T          @�{�5�@\@ ��A�{B��f�5�@�(�@��B(�B�8R                                    Bx^m_b  
�          @����,��@�=q@c�
A�G�B��,��@���@�\)B<G�B�aH                                    Bx^mn  �          @���'�@��@.{A���B�(��'�@�\)@��\B!
=B��f                                    Bx^m|�  "          @�R�@��@��H?�@�Q�B�=�@��@���@%�A���B��f                                    Bx^m�T  �          @�{�a�@�{>�p�@>�RB�p��a�@��R@�HA��RB�                                     Bx^m��  �          @��H��@��
?�G�Ae�B�B��H��@�@l��A�\)B�aH                                    Bx^m��  
�          @�{�9��@�33?���AK33B�#��9��@��R@fffA�B���                                    Bx^m�F  
�          @���ff@�\)?(��@��HC\��ff@�ff@��A��C�3                                    Bx^m��  �          @��
���@[�?�ffA-�C�����@3�
@�A�{Ch�                                    Bx^mԒ  
�          @�����@���?�ffA((�C5�����@�=q@9��A���C
��                                    Bx^m�8  
�          @�����\@�{���
�$z�C
�3���\@���?�  A"=qC�                                    Bx^m��  
�          @�ff�6ff@У�?W
=@�Q�B�B��6ff@��
@<��A\B���                                    Bx^n �  #          @�녿�z�@��?�R@�(�B�ff��z�@θR@:=qA���B�                                    Bx^n*  �          @�Q���@��>�ff@dz�B����@���@*=qA��B���                                    Bx^n�  �          @�G����@���?(�@�=qB�ff���@љ�@;�A��B�8R                                    Bx^n,v  
�          @�p��C�
@�{>8Q�?�
=B��
�C�
@���@�A�33B��)                                    Bx^n;  "          @����a�@�(�>L��?�\)B����a�@�\)@��A��RB�=q                                    Bx^nI�  "          @�z��q�@�  ��
=�XQ�B�aH�q�@��?���AK�B�{                                    Bx^nXh  �          @�(����@�Q�?��@�Q�C L����@���@z�A��C�H                                    Bx^ng  
(          @����J=q@��>��@UB�8R�J=q@��\@(�A�B��                                    Bx^nu�  
�          @�p��\(�@�  >8Q�?���B��3�\(�@�33@��A��B��                                    Bx^n�Z  	�          @�{�33@љ�?�
=A�\B��33@��@O\)A�33B�B�                                    Bx^n�   
Z          @����:�H@�33?�G�A$  B�\)�:�H@�33@P  A��B�\                                    Bx^n��  �          @�p��B�\@���@7�A��B����B�\@�(�@���B*�B���                                    Bx^n�L  �          @�{��\)@�33@U�A݅B�B���\)@�@�ffB:  B���                                    Bx^n��  
�          @�{���H@�33@333A�B�p����H@��H@���B&�B�ff                                    Bx^n͘  �          @�R�+�@��@7
=A��B��R�+�@���@�=qB(=qB                                     Bx^n�>  T          @�{��@�  @;�A�\)B�
=��@�
=@�33B+
=B�z�                                    Bx^n��  
�          @�>�G�@Ϯ@@  A�Q�B�B�>�G�@�{@�p�B-�B�Q�                                    Bx^n��  U          @�
==��
@�=q@W�A���B��==��
@��@��RB:�
B��                                    Bx^o0  �          @�
=�W
=@��
@6ffA��B��\�W
=@��
@��B'(�B�ff                                    Bx^o�  "          @�
=��\)@�ff@*=qA�\)B��쾏\)@�Q�@��B �RB��H                                    Bx^o%|  T          @�ff��@��@)��A�33B��)��@�\)@�(�B G�B�Ǯ                                    Bx^o4"  �          @�R��Q�@��H@��A���B�\��Q�@�G�@�Q�B  B�#�                                    Bx^oB�  
�          @�R����@�p�?��RA�{B�aH����@�{@�=qB	�\B�G�                                    Bx^oQn  	�          @�R�!G�@��
@z�A��B�k��!G�@��
@�(�B��B�.                                    Bx^o`  
�          @�\)>�@�\)?�Al(�B��>�@�G�@|(�B\)B���                                    Bx^on�  "          @�ff���@�(�?�Ax��B�=q���@�@~�RBG�B��                                    Bx^o}`  
�          @�{�W
=@�@�RA��
B�W
�W
=@�=q@�ffB=qB�(�                                    Bx^o�  
�          @�
=���@��?��HA}G�B�uÿ��@�33@\)B�B��)                                    Bx^o��  �          @�  ��(�@�p�?��
AC�
B�𤿼(�@�33@g�A�B�\                                    Bx^o�R  "          @���  @�ff?\AB�HB�Q쿠  @�(�@g
=A��B���                                    Bx^o��  
�          @��\@�\)?�ffAffB̅�\@��@J�HA�=qB�
=                                    Bx^oƞ  "          @�R�z�@�\)?�{AG�B�  �z�@���@HQ�A�z�B��H                                    Bx^o�D  
�          @�R�G�@�G�?B�\@\B����G�@�
=@4z�A��RB��                                    Bx^o��  "          @��ÿ�p�@��
�E��\BǞ���p�@�Q�?��HA9�B��                                    Bx^o�  �          @�  ���\@�녿s33��B�ff���\@�Q�?��\A!G�Bȏ\                                    Bx^p6  �          @��p��@��?�(�Amp�Bģ׿p��@�=q@fffB  B��                                    Bx^p�  
�          @�녿��H@�33@dz�A���B�  ���H@�\)@�{B?�B��                                    Bx^p�  �          @��ÿ�G�@�33@AG�A�\)B�33��G�@�(�@�\)B)z�B۞�                                    Bx^p-(  "          @߮���R@Å@?\)A�G�B�aH���R@��@��RB*33Bսq                                    Bx^p;�  
�          @�  ���@�  @,��A�{BϽq���@�(�@�\)Bp�B�u�                                    Bx^pJt  �          @�녿Y��@�\)@%A���B���Y��@�z�@�B\)B���                                    Bx^pY  �          @�녿�ff@љ�?�33Ay�B����ff@��@r�\B�B�W
                                    Bx^pg�  �          @�����@�?��Ay��B�����@��@o\)B �B���                                    Bx^pvf  �          @�  �p�@�33?�(�A�z�B�k��p�@��R@r�\B  B��                                    Bx^p�  "          @�ff�.{@�ff?�  AG�B�p��.{@��R@R�\A�RB�                                    Bx^p��  
�          @�(��\)@ȣ�?��A  B�W
�\)@�p�@7�A�
=B�p�                                    Bx^p�X  �          @����'�@�ff?�\)A8Q�B��
�'�@�  @J=qA�p�B��                                    Bx^p��  "          @��'�@���?�\AmG�B�.�'�@��H@a�A�33B�L�                                    Bx^p��  
Z          @�ff��@ȣ�?��RA�B�W
��@�z�@qG�B
=B��f                                    Bx^p�J  
Z          @�ff�z�@���@��A��B�=q�z�@�
=@{�B

=B��
                                    Bx^p��  �          @���\@ə�?�A��B�����\@�ff@mp�B  B���                                    Bx^p�  �          @��E�@�=q?�
=AB����E�@�{@:�HA�ffB���                                    Bx^p�<  �          @�
=�XQ�@�=q>�G�@i��B�G��XQ�@��@p�A�Q�B�                                    Bx^q�  
(          @�ff�>�R@�  ?z�@�G�B�
=�>�R@���@��A�33B�k�                                    Bx^q�  "          @��C33@ƸR>�?��B�ff�C33@���?�Q�A�z�B�                                    Bx^q&.  �          @����R@��u�G�B�33��R@�
=?�33A]��B߀                                     Bx^q4�  T          @����@�
=��\)�
=B�L���@�ff?���Aup�B��)                                    Bx^qCz  "          @ۅ�0  @���=�\)?!G�B�aH�0  @�\)?�33A���B�p�                                    Bx^qR   "          @�p��/\)@��>���@S33B�  �/\)@�p�@p�A��B�R                                    Bx^q`�  �          @ۅ�R�\@�  >B�\?˅B�z��R�\@�{?�A�z�B��                                    Bx^qol  �          @��H�Z�H@��>�ff@q�B��\�Z�H@�\)@Q�A���B���                                    Bx^q~  �          @��
���ÿE�@��B$C=����ÿ��H@uBffCJE                                    Bx^q��  
�          @�\)��z�>\@w�BQ�C/����z�@  @s�
B
=C;��                                    Bx^q�^  �          @�Q���z�>Ǯ@�B
=C/�H��z�Q�@��BG�C=�                                    Bx^q�  T          @�z���Q�?���@y��B	��C&33��Q�>8Q�@��
B(�C2!H                                    Bx^q��  "          @�(�����@p  @#�
A�p�C:�����@7�@aG�A��RCJ=                                    Bx^q�P  �          @���
=@��H?���A�
=C�f��
=@W�@B�\A�Q�C.                                    Bx^q��  �          @�p���@�{?ٙ�Adz�C �{��@�  @FffA�z�C��                                    Bx^q�  �          @���  @���?�A���B����  @�G�@UA�\)Cff                                    Bx^q�B  
�          @�ff���\@���Q����C�{���\@�Q�?0��@���Cz�                                    Bx^r�  T          @�z��tz�@�ff�L����33B����tz�@�
=?333@�z�B�z�                                    Bx^r�  "          @߮�A�@���@@  A�33B�W
�A�@�
=@�B�\B�33                                    Bx^r4  �          @���333@�\)@R�\A�\)B�8R�333@��\@�ffB)=qB���                                    Bx^r-�  
�          @�Q��ff@��H@��B%B׊=��ff@X��@�p�Bb�B�(�                                    Bx^r<�  �          @�=q��
@��@n{B z�Bܮ��
@���@��
B<G�B�8R                                    Bx^rK&  "          @�=q�%�@��R@I��AՅB䙚�%�@�33@��B$��B�u�                                    Bx^rY�  "          @��
�?\)@�p�@B�\A���B��?\)@��@�  B=qB��f                                    Bx^rhr  T          @�33�j=q@�33@�A��B�W
�j=q@�\)@tz�BQ�B��                                     Bx^rw  T          @��H��Q�@�33?�=qAo�C ����Q�@�(�@N�RAمC��                                    Bx^r��  
�          @��
��=q@�?aG�@�\C����=q@��@��A��C
p�                                    Bx^r�d  �          @�(����@��
?#�
@��C�����@��@G�A��C��                                    Bx^r�
  T          @�����@��>�@w�CaH����@��?�33Aw�C@                                     Bx^r��  T          @����@��R>���@��C� ���@�?��HA_�CT{                                    Bx^r�V  �          @��H��z�@�(������\C���z�@�\)?��HA��C	h�                                    Bx^r��  �          @�(����@��H��\)�\)Cff���@�\)?�{A�
C\                                    Bx^rݢ  
�          @�\��ff@����B�\���
Cu���ff@��?�Q�AffC@                                     Bx^r�H  
�          @���Q�@��þ.{����C�f��Q�@�z�?���A33C�R                                    Bx^r��  
(          @��
��z�@�(��
=q��z�C� ��z�@�33?J=q@�(�C�3                                    Bx^s	�  
�          @��
����@tz��U���Q�C� ����@�=q����=qC	��                                    Bx^s:  �          @�33��{@xQ��j=q��z�Cff��{@�
=�\)��Q�C��                                    Bx^s&�  
�          @�33���@�Q��333��ffC�����@���У��T��C	�3                                    Bx^s5�  
�          @ᙚ��\)@p�������
=C�\��\)@�Q쿨���+�C!H                                    Bx^sD,  T          @�  ����@s33�:�H����C�)����@�{��ff�p��C
��                                    Bx^sR�  T          @ᙚ��G�@�=q����33Ch���G�@���xQ����RC�                                    Bx^sax  �          @�G����@�
=>�33@7
=C ޸���@�?��
Aj�RCc�                                    Bx^sp  �          @�\�|(�@�33>Ǯ@L(�B��
�|(�@�G�?�A|Q�B�                                    Bx^s~�  
�          @�\���\@�녿z����B������\@���?h��@�33C #�                                    Bx^s�j  
�          @ᙚ��(�@����(�� (�Cٚ��(�@���>.{?��C
                                    Bx^s�  �          @߮���@����.{���CL����@�=q��Q��?\)C                                      Bx^s��  
�          @�(���ff@�������C�)��ff@���?G�@���C��                                    Bx^s�\  
�          @ٙ����@���{���
C
0����@�p������\Cn                                    Bx^s�  �          @ٙ����@�Q��Q���p�C�H���@���5�\C�                                     Bx^s֨  
�          @�(����
@�?��A\z�B��f���
@�G�@O\)A��B��                                    Bx^s�N  T          @�z��{@�z�?�
=Ab{B�k���{@�  @P��A�
=B��
                                    Bx^s��  "          @�(��"�\@�G�?���A  B���"�\@�G�@-p�A���B�Q�                                    Bx^t�  
�          @��
��R@�=q?��AN�RB��H��R@��R@FffA�z�Bޞ�                                    Bx^t@  "          @�z���H@��H?���Au�B�\)���H@�p�@XQ�A�G�B�(�                                    Bx^t�  
�          @�z��@�G�?���Ay�B����@��
@X��A�B��f                                    Bx^t.�  "          @�33��33@���?�
=A�33Bՙ���33@��H@]p�A�
=B�z�                                    Bx^t=2  �          @�(���33@У�?^�R@�=qBԏ\��33@��H@\)A���BָR                                    Bx^tK�  T          @ۅ��Q�@���?@  @�G�B�uÿ�Q�@�  @��A�=qB�                                      Bx^tZ~  T          @�33��\)@Ӆ?��A�\B�z`\)@�z�@,(�A�
=B�B�                                    Bx^ti$  �          @ۅ��{@�=q?�\)A8��B�W
��{@���@>{A��
B�k�                                    Bx^tw�  "          @��H�޸R@�ff?�(�AG33B�8R�޸R@��
@B�\A��HB�                                    Bx^t�p  �          @��H��
@��H?�{A��B����
@��
@)��A���B��f                                    Bx^t�  "          @ٙ���@�Q�?޸RAnffB�#׿�@�(�@O\)A�z�Bٮ                                    Bx^t��  �          @�G��ٙ�@�33?У�A^=qB��H�ٙ�@�  @I��A�\)B��
                                    Bx^t�b  
�          @ٙ��	��@�33?���A��B�W
�	��@�z�@&ffA�=qB�
=                                    Bx^t�  
�          @׮�!�@�33?&ff@��
B�  �!�@�  @Q�A��B�k�                                    Bx^tϮ  "          @׮�=q@ȣ�>���@XQ�B�  �=q@�\)?�A���B��
                                    Bx^t�T  
�          @�
=�(�@\?��
A
=B�Ǯ�(�@�z�@�RA��B���                                    Bx^t��  	�          @׮�z�@���?�33A@Q�B�\)�z�@�(�@6ffA�
=B��)                                    Bx^t��  T          @���� ��@���?�(�Al��B�G�� ��@�p�@HQ�A�\)B�3                                    Bx^u
F  
�          @ٙ��
=q@�
=?�33AaB�Q��
=q@�(�@G
=A�{B���                                    Bx^u�  
(          @�G���@��@P  A�33B��
��@�G�@�
=B#��B�\)                                    Bx^u'�  T          @�Q��
�H@�G�@L(�A�B����
�H@��@�p�B#�HB�p�                                    Bx^u68  
�          @���#33@�33?�ffAS
=B�p��#33@���@=p�A��B�k�                                    Bx^uD�  
�          @�  �(Q�@�@�A�(�B�R�(Q�@�ff@j=qB��B�                                      Bx^uS�  
�          @�  �)��@��@z�A��B噚�)��@���@g�B��B��                                    Bx^ub*  �          @׮�$z�@���@G�A��
B�{�$z�@�=q@dz�B �B��
                                    Bx^up�  T          @�Q���\@���?��HA�33B݅��\@�z�@U�A���B��                                    Bx^uv  
�          @أ��333@�ff?ǮAUG�B�ff�333@�p�@:�HA�z�B�                                    Bx^u�  �          @�Q��7�@�?��
AP��B�3�7�@���@8Q�A�p�B���                                    Bx^u��  
�          @ٙ��9��@�=q?�  A33B��9��@��@��A��RB�#�                                    Bx^u�h  "          @�Q��I��@�z�?�ffAxz�B�k��I��@�=q@C�
A�\)B���                                    Bx^u�  T          @�\)�;�@�p�?�Q�A#
=B�3�;�@��@!�A�G�B�L�                                    Bx^uȴ  T          @�\)�.�R@��H?Y��@�
=B�W
�.�R@�\)@�RA��B�                                    Bx^u�Z  T          @�=q��  @У׿����(�B�  ��  @�
=?xQ�AQ�B�8R                                    Bx^u�   
�          @׮��33@�ff�8Q쿾�RB��)��33@�=q?��A4(�BՀ                                     Bx^u��  T          @�
=�7�@�33?У�A`Q�B�33�7�@�=q@;�AΣ�B왚                                    Bx^vL  
�          @�\)�+�@���?���A$��B���+�@��H@"�\A�ffB�.                                    Bx^v�  
�          @׮�33@�(�?&ff@���Bר��33@�=q@A�Q�B�ff                                    Bx^v �  
�          @�p���\@�z�?�\@��B�\��\@�33?���A�Q�B�k�                                    Bx^v/>  �          @�{���@�G�>W
=?��B�#׿��@�=q?�
=Ak33Bƽq                                    Bx^v=�  "          @أ׾��
@ָR��Q��G
=B�zᾣ�
@��
?�z�A�\B��\                                    Bx^vL�  �          @�
=>B�\@��?O\)@�B��3>B�\@��@�\A�\)B��                                     Bx^v[0  
�          @ָR�z�@�z�?:�H@�\)B�녿z�@��@��A���B�p�                                    Bx^vi�  T          @�
=�@  @��
?aG�@���B��H�@  @�Q�@A��RB���                                    Bx^vx|  
�          @�\)�fff@��H?uA33BÏ\�fff@�
=@��A�p�BĀ                                     Bx^v�"  
�          @�
=��
=@љ�?c�
@���B�\)��
=@�{@z�A�  BɅ                                    Bx^v��  T          @ָR��33@�G�?L��@�z�B��)��33@�ff@�RA��B��                                    Bx^v�n  T          @�Q�Tz�@�z�?L��@�=qB�8R�Tz�@ə�@\)A��B���                                    Bx^v�  
�          @�ff�У�@�(�?�z�A�
BШ��У�@�
=@!�A�  BҊ=                                    Bx^v��  T          @ָR��p�@ʏ\?��A8��Bҏ\��p�@�(�@,(�A��B�                                    Bx^v�`  "          @�ff��@�33?���A�RB��H��@��R@�HA��B���                                    Bx^v�  T          @ָR����@�z�?\(�@�33B�Q����@���@�RA���B�
=                                    Bx^v��  
�          @�{��@�33?�  A,(�B�z��@�@%A�z�B�z�                                    Bx^v�R  �          @�p��33@�\)?�z�A (�B؅�33@��\@{A�Q�B���                                    Bx^w
�  T          @�z��\)@�(�?��A(�B�#��\)@�Q�@��A���Bފ=                                    Bx^w�  �          @�z��p�@�p�?��A�
B�W
�p�@��@�A��RBݙ�                                    Bx^w(D  "          @������@�G�?�
=A#�
B�Ǯ����@�z�@   A�=qB��)                                    Bx^w6�  "          @�z��p�@�33?O\)@��B�uÿ�p�@�G�@��A��RB���                                    Bx^wE�  
�          @��Ϳ��@˅?333@��B�aH���@��@�A�z�B���                                    Bx^wT6  T          @�\)��@��H?��A�RB�녿�@�
=@Q�A���B�Ǯ                                    Bx^wb�  
Z          @�  ��
=@ə�?�{A:�RB���
=@��
@)��A��B�Q�                                    Bx^wq�  
�          @��ÿ�33@Ϯ?#�
@�B�p���33@�
=?�p�A��BѨ�                                    Bx^w�(  
�          @�G��z�H@�>�G�@o\)BĮ�z�H@�{?�=qAzffB�L�                                    Bx^w��  
�          @ٙ���\)@�>�33@;�B����\)@θR?�p�Al(�BǸR                                    Bx^w�t  "          @�G�����@��?�\@�33B�B�����@��?��A�
=B�                                      Bx^w�  
�          @�  �(�@�33?���A
=B��=�(�@�\)@{A�ffB�.                                    Bx^w��  �          @�Q�>�
=@�{�
=q���B��)>�
=@��?Y��@�B��
                                    Bx^w�f  
Z          @�\)��@���k����\B����@�ff>��@\)B��                                    Bx^w�  T          @أ׾�ff@��n{��p�B��R��ff@�\)>�ff@u�B��                                    Bx^w�  
�          @ָR��@��;����Q�B�L;�@Ӆ?fff@�ffB�W
                                    Bx^w�X  "          @׮�+�@�ff>�=q@�
B�W
�+�@�  ?�\)A_
=B��                                    Bx^x�  �          @�
=��ff@�\)?E�@ҏ\B����ff@�ff@�
A�33B�                                    Bx^x�  "          @�ff��\@ə�?�  A	B����\@�
=@  A���B�Ǯ                                    Bx^x!J  T          @�p���{@�(�>��@G�BԀ ��{@�ff?��AU��B�ff                                    Bx^x/�  "          @���G�@���?W
=@��B܊=�G�@��@33A�Q�B�aH                                    Bx^x>�  �          @�(��>{@�(�>�\)@=qB�=�>{@��R?�Q�AJ{B��                                    Bx^xM<  
�          @�(��dz�@����\�UB�p��dz�@���?8Q�@���B�                                    Bx^x[�  "          @��r�\@��ͿE���B�8R�r�\@�{>��R@)��B�                                    Bx^xj�  �          @�p��^{@��H����\)B��^{@�  ?xQ�A(�B�u�                                    Bx^xy.  �          @أ���R@��>u@��B�L���R@���?���AI�B�u�                                    Bx^x��  "          @����$z�@��@z�A��\B��$z�@�z�@L(�A�z�B���                                    Bx^x�z  
(          @׮�5@���?�A���B�(��5@��@@��Aԏ\B�8R                                    Bx^x�   
(          @ָR�?\)@�33@)��A���B�Q��?\)@��@hQ�B�HB�p�                                    Bx^x��  
�          @��2�\@���@ffA�p�B�=�2�\@�z�@I��A�z�B���                                    Bx^x�l  
�          @��/\)@�{@.�RA���B�z��/\)@��@n{B�\B�L�                                    Bx^x�  T          @�{�p�@��R@p�A�
=B�{�p�@�(�@`��A��B�.                                    Bx^x߸  #          @ָR�&ff@��H?���A�B��f�&ff@��
@A�AׅB��                                    Bx^x�^  
�          @أ�� ��@�G�?�  Ap  B�33� ��@��H@7�A�z�B�aH                                    Bx^x�  T          @׮��R@�?�
=ADQ�B۽q��R@���@$z�A��B��                                    Bx^y�  �          @�\)��@ƸR?��
A0Q�Bڽq��@��@�A��
B��H                                    Bx^yP  �          @�  �	��@ə�?�=qA33Bٞ��	��@��@\)A�\)B�u�                                    Bx^y(�  "          @׮�;�@��R?��A��B�k��;�@���@
=qA�B��
                                    Bx^y7�  
�          @����,��@�33?���AB��,��@�G�@{A��B��                                    Bx^yFB  "          @�G��
=@���?G�@ҏ\B؀ �
=@�z�?���A�B��                                    Bx^yT�  "          @�Q��	��@���?�p�A'�Bٽq�	��@�ff@�A�=qB۳3                                    Bx^yc�  "          @ָR�p�@�Q�?��@��B��)�p�@���?޸RAr=qB�(�                                    Bx^yr4  �          @��
�L(�@�Q쾏\)�{B���L(�@�
=?G�@أ�B�Q�                                    Bx^y��  �          @Ӆ����@��H����p�C�=����@����  �  C G�                                    Bx^y��  �          @�����@�33�8Q����C�R���@�������C�3                                    Bx^y�&  T          @�33���@XQ��R�\���CǮ���@z�H�'���G�C��                                    Bx^y��  "          @Ӆ��\)@q��8�����
Cٚ��\)@���
=q��Q�C�
                                    Bx^y�r  �          @�33�1�@�
=    =#�
B��H�1�@��
?�=qA�\B晚                                    Bx^y�  "          @��
�@  @�����
�\)B�L��@  @�Q�?��
Az�B�
=                                    Bx^yؾ  �          @�=q���@��ÿ�(��T(�C�����@�\)�#�
���C^�                                    Bx^y�d  �          @�G����@�녿z����\CY����@��H>u@Q�C(�                                    Bx^y�
  "          @������\@��þ�
=�mp�C.���\@���>�(�@vffC0�                                    Bx^z�  "          @���>{@��R?�
=A%��B����>{@�p�@
=qA�ffB�                                    Bx^zV  �          @�=q�A�@�\)?��\AffB��A�@��R@   A�Q�B�
=                                    Bx^z!�  T          @�=q�g�@��H?�z�A"�HB�(��g�@��@z�A��B��                                    Bx^z0�  �          @�G��e�@�=q?c�
@�(�B����e�@��\?�ffA�
=B�ff                                    Bx^z?H  "          @�=q�fff@���?��A\(�B���fff@���@=qA���B��                                    Bx^zM�  
�          @ҏ\���\@���?��A{C(����\@�Q�?��A�{C��                                    Bx^z\�  �          @�=q��z�@���?��
A4��C0���z�@�33@ffA�{C�                                    Bx^zk:  "          @�33���@�p�?���A�ffCG����@�G�@%A��C�3                                    Bx^zy�  �          @�����33@�33��=q�$  CB���33@�
=���
�C33Cz�                                    Bx^z��  T          @Ӆ�n{@����
=��=qC0��n{@����=q�k
=C �                                    Bx^z�,  �          @�{�z�H@�G��J=q�ٙ�B�  �z�H@�33=�?��B�k�                                    Bx^z��  �          @׮����@�p���33����CQ�����@�p�����z�B��H                                    Bx^z�x  �          @ָR��  @�녾���p�C  ��  @���?��@�C(�                                    Bx^z�  "          @ָR�xQ�@�녾���G�B�aH�xQ�@���?!G�@��B��3                                    Bx^z��  "          @ָR�s�
@��>�@�{B��s�
@��H?��
A<(�C 33                                    Bx^z�j  �          @أ׿�\)@��@p�A���B�LͿ�\)@��
@K�A�(�B��H                                    Bx^z�  �          @أ�� ��@���@G�A��B��� ��@��@=p�A��B�33                                    Bx^z��  �          @ڏ\��@�=q?�{A[�Bսq��@�
=@(Q�A�  Bי�                                    Bx^{\  �          @ٙ��G�@�p�?n{@���B뙚�G�@�{?��A��B�                                     Bx^{  
�          @���r�\@�녾k���Q�B����r�\@���?.{@���B���                                    Bx^{)�  T          @љ���(�?z�H?aG�@�Q�C+E��(�?Q�?��
A�C,��                                    Bx^{8N  T          @ҏ\��33?��E���\)C'Y���33?��
�����
C&s3                                    Bx^{F�  �          @Ӆ�ƸR?�\)��(��*�RC#@ �ƸR@�\�h����z�C!�\                                    Bx^{U�  
�          @У���{?���R��ffC#����{?�녾��
�3�
C#�                                    Bx^{d@  T          @Ϯ���?녿�
=�)p�C.�
���?@  ��=q�=qC-5�                                    Bx^{r�  �          @�\)�˅��Q�G��޸RC7B��˅�k��Tz���z�C6)                                    Bx^{��  �          @�G���G�?fff��
=��ffC+����G�?�Q��G���  C(�
                                    Bx^{�2  T          @�G����@G�?У�Av{C {���?�p�?�A�(�C"�                                    Bx^{��  �          @Ӆ��Q�@@	��A���Cٚ��Q�?�p�@\)A�Q�C!�                                    Bx^{�~  �          @�G��n�R@ ��@��BF�\C�H�n�R?��H@��BTC"�                                    Bx^{�$  �          @�z����H?���@�z�B>�C{���H?��
@��BI�
C%��                                    Bx^{��  
Z          @�p����?��
@fffB�C!p����?��H@u�B�\C'5�                                    Bx^{�p  T          @ָR����@$z�@mp�BffCW
����?�(�@��BG�C#�                                    Bx^{�  T          @�ff��ff?��
?��AT(�C*���ff?G�?�
=Ag�
C-�                                    Bx^{��  
�          @�
=�˅?�=q?��A5G�C&��˅?�{?��AT��C'�                                    Bx^|b  �          @�{���@?�A�(�C �q���?��
@��A��C#��                                    Bx^|  �          @�
=��z�@�@��A�C�\��z�?�@!G�A�33C!��                                    Bx^|"�  �          @�(���Q�@>{@�A�\)C����Q�@'�@"�\A��RC��                                    Bx^|1T  "          @������R@33@��A��C\���R?�z�@0��A�\)C!��                                    Bx^|?�  �          @�p���  @4z�@W
=A���C�=��  @�\@o\)B	
=Cc�                                    Bx^|N�  T          @�{��z�@,��@��A��Cn��z�@�@(��A�
=C�=                                    Bx^|]F  T          @�{��(�@8��?aG�@�(�C�)��(�@.�R?�ffA4��C&f                                    Bx^|k�  �          @�{��G�@Fff@�\A���C�=��G�@0��@�RA��\C�                                     Bx^|z�  
Z          @�z�����?   @C�
A�C/
����=#�
@FffA��HC3��                                    Bx^|�8  "          @�  ��\)?J=q?��
AZ�HC,�\��\)?��?У�AiG�C.�                                    Bx^|��  T          @����Q�@e��   ���
C����Q�@z=q�������HC\)                                    Bx^|��  T          @�=q���@��R�\)��z�C	=q���@�������`��Ch�                                    Bx^|�*  �          @�p���G�@N�R�L�;��C�\��G�@Mp�>�
=@hQ�C                                    Bx^|��  
�          @�\)��p�@aG��(���ffC0���p�@dz���ͿQ�C��                                    Bx^|�v  
�          @������R@�p������\)C
L����R@�ff=�Q�?B�\C

                                    Bx^|�  �          @�G���G�@��
�����p�C��G�@�\)��(��g�C\)                                    Bx^|��  T          @ٙ���
=@��ÿ����>ffC���
=@��0����z�C0�                                    Bx^|�h  �          @�33�5�@��\��  �z�B�u��5�@�33�Tz�����B��                                    Bx^}  
�          @�z��Q�@�{�\(���B��Q�@��
�(����
=B��H                                    Bx^}�  
�          @�p���
=@����Q��;(�B�\��
=@�
=���\��HBͅ                                    Bx^}*Z  
�          @��j=q@�=q�5�ÅB����j=q@�p��
=����B�{                                    Bx^}9   
�          @�{�U�@�z�Ǯ�V�\B�(��U�@����B�\��Q�B�q                                    Bx^}G�  T          @ۅ�c33@�녿u���B�8R�c33@�z�B�\����B�z�                                    Bx^}VL  
�          @�Q�����@S�
@HQ�A��C�R����@6ff@c�
B \)C�
                                    Bx^}d�  
�          @ٙ����@Q�@g
=BffC�����@0��@���B�\CY�                                    Bx^}s�  �          @������R@e@O\)A�C�����R@G
=@l��B{C��                                    Bx^}�>  �          @�G���=q@e�@fffA�Q�C���=q@C�
@���B�C5�                                    Bx^}��  
�          @ٙ���{@���@J�HA��C	����{@c33@k�B�RC^�                                    Bx^}��  S          @��H���
@��\@#�
A�(�C�����
@�{@I��A�  C(�                                    Bx^}�0  �          @����Q�@�=q@)��A�33C����Q�@�p�@QG�A�CE                                    Bx^}��  T          @���y��@�Q�@q�B33C&f�y��@^{@���Bp�C
G�                                    Bx^}�|  "          @ۅ�qG�@��@0��A���B���qG�@��\@Y��A�RCs3                                    Bx^}�"  �          @������@�ff@��A��HC8R����@��@7
=A��C^�                                    Bx^}��  
�          @�ff�e@�33�S33��p�B���e@���(Q���{B��q                                    Bx^}�n  �          @������@�
=��=q�~�HC0�����@�p���z�� ��C �                                    Bx^~  
Z          @��@  @�ff@
=A�33B� �@  @�(�@5A��B�Q�                                    Bx^~�  	�          @�33�b�\@�z�@33A�
=B��{�b�\@��\@.�RA�z�B�                                    Bx^~#`  �          @��
����@���?��A:ffC� ����@��H?���A��\C��                                    Bx^~2  
�          @��
��=q@>�R�0  ��z�C�R��=q@S�
�ff��Q�C
                                    Bx^~@�  "          @�����(�@}p�=u?   C�{��(�@z�H?�@�\)C�                                    Bx^~OR  �          @�ff�#�
@���@o\)B{B�\�#�
@�Q�@�33B�B�B�                                    Bx^~]�  �          @�ff�ff@���@��B3p�B�G��ff@h��@�  BM=qB��                                    Bx^~l�  
Z          @�  ����@��@���B#ffBܣ׿���@���@��B>ffB��)                                    Bx^~{D  
�          @�Q�h��@�
=@�  BD  B�{�h��@qG�@���B`=qB�.                                    Bx^~��  �          @�녽�G�@�ff@�{BJB��ͽ�G�@o\)@��RBg�B�W
                                    Bx^~��  T          @�p����@���@�33BS=qB�(����@U�@��Bo�B��H                                    Bx^~�6  �          @�z��33@�z�@l��B	z�B�W
�33@�z�@���B#
=B�8R                                    Bx^~��  �          @ָR��
@��R@U�A�z�B�  ��
@�Q�@z�HB��B�L�                                    Bx^~Ă  �          @�=q��{@�=q�\)����C��{@��H��
=���HC                                      Bx^~�(  �          @�{��(�@��R�   ���
C{��(�@�\)��
=��(�CaH                                    Bx^~��  "          @����z�@~{�(����
=CO\��z�@�Q������  CQ�                                    Bx^~�t  
�          @�ff���@Z=q�|���	�C�����@w
=�`����
=CW
                                    Bx^~�  T          @�  ��  @<����{�33C}q��  @[��s33� G�C�=                                    Bx^�  "          @��H���R@A���Q��+��C�����R@g
=����{C�                                    Bx^f            @�\)��  @=q��=q��HCB���  @<(���  ��HC��                                   Bx^+              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^9�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^HX              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^V�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^e�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^tJ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�<              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�.              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�z              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�l              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�$              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�2�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�A^              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�P              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�^�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�mP              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�{�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^��B              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^��4              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^��&              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�r              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�  �          @���xQ�@>{@���Bm��B��xQ�@=q@��B��B߮                                    Bx^�+�  
Z          @�
=���@~�R@�33B7��B�33���@aG�@��BN�Bݽq                                    Bx^�:d  T          @����Q�@��\@(��A��B�33�Q�@���@L(�A��HB�                                    Bx^�I
  �          @�
=�/\)@��R@�A��B�8R�/\)@�
=@)��A���B�.                                    Bx^�W�  T          @�Q��*�H@�=q@�A�=qB�Q��*�H@��\@*=qA�ffB�33                                    Bx^�fV  �          @أ��(Q�@��@I��A�ffB�Q��(Q�@�Q�@j�HB�
B�ff                                    Bx^�t�  T          @�Q��%�@���@8��Aʣ�B�  �%�@��R@[�A�=qB��                                    Bx^���  T          @ڏ\�/\)@�=q@��A��B�ff�/\)@��\@-p�A��\B�Q�                                    Bx^��H  �          @��
�S�
@��R?�ffAt(�B�=q�S�
@�  @
=A��B��                                    Bx^���  �          @��
�:=q@���@G�A���B���:=q@�Q�@5�A�33B�q                                    Bx^���  �          @�p��6ff@�ff@33A�  B�G��6ff@�
=@(Q�A��RB�{                                    Bx^��:  �          @�
=�C�
@�G�?�p�AD��B����C�
@��
@z�A��RB�(�                                    Bx^���  �          @�ff�N{@��R?��A9B�q�N{@���?�(�A�ffB�{                                    Bx^�ۆ  �          @޸R�P  @�  ?�p�A$  B��)�P  @�33?�Ar�RB�{                                    Bx^��,  �          @�(��A�@��H?xQ�Ap�B��H�A�@�
=?ǮAQG�B���                                    Bx^���  �          @�z��HQ�@\?Q�@��B� �HQ�@�
=?�33A<  B�W
                                    Bx^�x  �          @���G�@���=L��>�(�B����G�@��
?&ff@��
B�{                                    Bx^�  �          @�z��Dz�@��
����Q�B�=q�Dz�@�(�=�G�?k�B��                                    Bx^�$�  �          @�(��j�H@���Q��ڏ\B�G��j�H@�G�����(�B���                                    Bx^�3j  �          @�z��R�\@�\)�h����33B���R�\@�G���{�333B�33                                    Bx^�B  T          @�p��h��@�녿(������B�#��h��@�33��G��fffB���                                    Bx^�P�  T          @�p��I��@������2�RB�p��I��@��H�B�\��33B�                                    Bx^�_\  �          @�z��HQ�@�G��&ff��p�B��
�HQ�@\��\)�(�B�{                                    Bx^�n  �          @ڏ\���\@�ff�;���\)C�����\@�ff�!G����HC0�                                    Bx^�|�  �          @����@�{�&ff��Q�C���@���
�H���C��                                    Bx^��N  �          @׮���@�33����9G�C �����@�ff�\(���(�B���                                    Bx^���  �          @����z�@���Ǯ�Yp�C����z�@����{�ffC\                                    Bx^���  T          @ָR��z�@�����H��{C����z�@����p���{CxR                                    Bx^��@  �          @�\)��=q@`���\(�����C���=q@s�
�Fff��=qC5�                                    Bx^���  "          @׮��p�@p  �@���ՙ�C@ ��p�@�Q��*=q����C\)                                    Bx^�Ԍ  �          @أ�����@P���`����=qC�f����@dz��L����C5�                                    Bx^��2  �          @�\)��@HQ��z�H�ffC�R��@^�R�g���HC�)                                    Bx^���  �          @�
=��  @;��\)�p�C���  @Q��l�����C��                                    Bx^� ~  	�          @�G���(�@q��%�����CǮ��(�@\)��R��=qC0�                                    Bx^�$  �          @��
����@�Q��   ��  C
p�����@�
=�Q���(�C�q                                    Bx^��  �          @�{��@����8���͙�C	����@�G�� ����\)C�f                                    Bx^�,p  �          @�{���@z=q�Dz���(�C
s3���@�p��-p���ffC��                                    Bx^�;  
�          @أ����@w
=�:�H��(�Cff���@�33�$z�����C
��                                    Bx^�I�  �          @�����@�녿�
=�g�C�����@�{��  �+
=CǮ                                    Bx^�Xb  �          @��H�}p�@�\)������
B����}p�@�Q�u�   B�W
                                    Bx^�g  �          @��ÿ�=q@�z�@
=A�Q�B�p���=q@�@(��A��
B�G�                                    Bx^�u�  �          @�(���G�@��H@�A�Q�B��
��G�@��
@.�RA��B�                                    Bx^��T  �          @ۅ���R@�ff@!�A�G�B�{���R@�ff@C�
A�Q�B�(�                                    Bx^���  �          @ۅ���\@���@>�RAϮB�B����\@�  @`  A�33B�.                                    Bx^���  T          @ָR��{@�ff@B�\A�\)B��ÿ�{@��@`  B=qB�.                                    Bx^��F  �          @У���(�@3�
�Z=q���CW
��(�@E�J=q��CǮ                                    Bx^���  �          @ҏ\��33?\���H�3p�C �3��33?�z���{�,(�C@                                     Bx^�͒  T          @�=q�ff��������t�C_ٚ�ff�޸R�\��CX�                                    Bx^��8  T          @�z�������ƸR�v�
Cc��녿�33����8RC[�)                                    Bx^���  �          @�R��\��G��ڏ\{CN@ ��\��
=��z�L�C?�)                                    Bx^���  �          @�G��  ����Q�CZ�\�  ��p����ǮCP��                                    Bx^�*  
�          @�  ���*=q��=q�s�Cd�����
=�����=C^8R                                    Bx^��  T          @��� �׿�z����fC_�H� �׿����=q  CU��                                    Bx^�%v  �          @�R���ÿ�ff�ٙ�=qC\xR���ÿxQ������qCP�                                    Bx^�4  �          @�\)�n{�Ǯ�߮��Co��n{�xQ����H�=Ca�q                                    Bx^�B�  �          @��Ϳu��
=��33u�CpW
�u�����޸R\)Cd��                                    Bx^�Qh  �          @�R����R�ڏ\�
C�XR����33�߮�C|G�                                    Bx^�`  �          @�p��z�H�+��У��Cy�R�z�H�Q���
=�=CuT{                                    Bx^�n�  �          @�(��#33�����=q�}Q�CX���#33��{�θR��CP
                                    Bx^�}Z  �          @��&ff�33��p��rz�C]�=�&ff���
���H�~=qCVs3                                    Bx^��   �          @�
=��(��5���~�Cr�H��(��33����G�Cmff                                    Bx^���  �          @�  �ٙ��>{���
�w  Cp+��ٙ�����33� Ck                                    Bx^��L  T          @�׿�{�@����=q�r��CnT{��{��R��G�L�Ci.                                    Bx^���  �          @���C�
���l=qCk����"�\����{G�Cf��                                    Bx^�Ƙ  "          @�  �ٙ��HQ���=q�rz�CqxR�ٙ��&ff�љ�L�Cl��                                    Bx^��>  �          @�ff�
=�W
=��  �b33Cm���
=�7
=��  �q��Ci�
                                    Bx^���  T          @�p��p��\(���(��\�RCmY��p��<����z��l{Ci@                                     Bx^��  �          @��=q�[������X
=Cj���=q�<����G��f�
Cf�q                                    Bx^�0  
�          @�
=�����o\)���H�X��Crz�����P����(��h�
Co�                                    Bx^��  �          @�{�/\)�6ff����`�Cb#��/\)�
=���
�m�C\��                                    Bx^�|  �          @�ff�z��`  ���H�Y33Cl���z��AG���33�h=qCh��                                    Bx^�-"  
�          @�\)����W
=����[�\Ci�3����8Q�����j
=Ce��                                    Bx^�;�  T          @�ff�/\)�fff���H�L
=Ch��/\)�H������ZG�Cdٚ                                    Bx^�Jn  
�          @�����XQ���33�Z��Cj� ����9����33�i�Cf�=                                    Bx^�Y  �          @�p��!G��e��z��Pz�Cj���!G��HQ�����_{Cg5�                                    Bx^�g�  �          @��!��\����  �U\)Ci�q�!��>�R��  �c�Ce��                                    Bx^�v`  "          @�(��n{�%���z��GCV���n{�����=q�Qz�CQ�                                    Bx^��  �          @����-p��I�������Y�Ce\)�-p��,(���Q��fffC`�=                                    Bx^���  �          @�{�333�Y������P�Cf}q�333�<(�����^
=Cbc�                                    Bx^��R  �          @ᙚ�  �J=q����b
=Cj���  �,(����H�pQ�Cf
=                                    Bx^���  T          @�Q��C�
�R�\��33�G�Cc��C�
�7
=���H�T��C_                                      Bx^���  "          @ᙚ�Q��qG����R�3=qCe�Q��W�����@z�Ca�{                                    Bx^��D  �          @߮�O\)�fff��Q��7�HCd  �O\)�L�������D�
C`�)                                    Bx^���  �          @�
=�Tz��@����G��F�HC^+��Tz��%��Q��RffCY�                                    Bx^��  "          @�p����
�*�H�����p�CQ�����
����H��CN�\                                    Bx^��6  �          @߮���H��=q��ff�E  CL����H�����\�K�\CG�                                    Bx^��  T          @޸R���>\)�����\)C2�H���>�ff�����p�C/xR                                    Bx^��  �          @�ff��p�?xQ��]p����
C*�3��p�?�p��W���\)C(:�                                    Bx^�&(  �          @޸R��\)>�(��{��	��C/�q��\)?:�H�x����
C,�                                     Bx^�4�  T          @�\)���H>�{�q����C0�����H?!G��p  �(�C-�)                                    Bx^�Ct  T          @ٙ�����?�
=�����
C&�����?����
=q���HC$��                                    Bx^�R  
�          @׮��(�@>{�z�H��C5���(�@B�\�@  ��
=C�3                                    Bx^�`�  "          @�G����@w
=>�(�@g
=C5����@tz�?8Q�@ÅC�                                    Bx^�of  �          @ۅ���@���?8Q�@���C����@��\?��
A
�RC�R                                    Bx^�~  �          @�=q���R@�p�?���A:�RCY����R@���?�Q�Af{C#�                                    Bx^���  
�          @ٙ���33@|(�@�A���CQ���33@qG�@�RA���C�
                                    Bx^��X  
�          @������@u@7
=A�33Cff���@g
=@H��A܏\C�                                    Bx^���  �          @ҏ\�{�@q�@j�HB�
C{�{�@_\)@|��B�C
Y�                                    Bx^���  "          @��H�l��@j�H@��BC8R�l��@Vff@��\B#�RC	�\                                    Bx^��J  �          @����@���@��B �B�\���@�ff@��RB1
=B���                                    Bx^���  T          @Ϯ�+�@j�H@��BP�BȽq�+�@R�\@�=qBa�B�
=                                    Bx^��  �          @��>��?�=q@�p�B�� B���>��?aG�@�  B��{B��                                    Bx^��<  "          @��H�G
=@z�@��BN�RC@ �G
=?���@�=qBXz�C�R                                    Bx^��  �          @�\)�?�(�@�(�B��{C���?��\@��B��{C�f                                    Bx^��  �          @�
=��=q?��@��HB���C#׿�=q?�Q�@��RB�\)C�                                    Bx^�.  "          @���<#�
@�\)@�=qBC  B���<#�
@vff@�(�BT33B���                                    Bx^�-�  �          @���?�z�@�p�@K�A�{B��)?�z�@�p�@dz�BffB�L�                                    Bx^�<z  �          @�(��5@�G�@G�A��HB�R�5@���@_\)B G�B�8R                                    Bx^�K   "          @�=q���@^{@��A�z�CxR���@Q�@(��A�ffC
=                                    Bx^�Y�  "          @�\)��
=@vff?�
=A�
=CE��
=@l��@p�A�
=Cn                                    Bx^�hl  "          @�ff��{@Vff��\)�(��Cff��{@U>8Q�?�=qCn                                    Bx^�w  "          @�
=����@Y��@4z�A���CY�����@K�@C�
A�\)C.                                    Bx^���  �          @��
�9��@��
@{�B\)B�=q�9��@��@��B!ffB��                                    Bx^��^  "          @����G�@�Q�@�Q�B7�B����G�@x��@��BGQ�B�R                                    Bx^��  	�          @ȣ׿
=@h��@���BV�HB�p��
=@P  @���Bg��Bȏ\                                    Bx^���  T          @�(��p�@hQ�@��HBHz�B��p�@P  @��HBVB�u�                                    Bx^��P  �          @�z���\@}p�@�p�B?p�B� ��\@e�@�ffBNffB�G�                                    Bx^���  
�          @Ӆ�R�\@XQ�@��
B1��CB��R�\@A�@��B=z�C	ff                                    Bx^�ݜ  �          @�33�g�@AG�@��
B1�RC8R�g�@*�H@��\B<(�C�f                                    Bx^��B  �          @��
�C�
@^�R@�B533CO\�C�
@HQ�@�BA��CaH                                    Bx^���  "          @Ϯ�8��@��\@R�\A�ffB����8��@�=q@g�BffB���                                    Bx^�	�  T          @�z��X��@A�@���B0  C
0��X��@,��@��B:�CxR                                    Bx^�4  
�          @���Z=q@@  @�Q�B3\)C
���Z=q@*=q@�
=B>(�C                                    Bx^�&�  �          @�33�P��@e�@���B&�C\)�P��@P  @���B2�RC�                                    Bx^�5�  �          @��
�P  @��H@��\B�HC ���P  @q�@��
B$�C��                                    Bx^�D&  "          @�Q��P��@��@J�HA�(�B���P��@��@`  B�B�#�                                    Bx^�R�  �          @�33�~�R@�@1�A�(�C�f�~�R@�
=@FffA�
=CT{                                    Bx^�ar  �          @У��w
=@�
=@-p�A�G�C���w
=@�Q�@B�\Aޏ\C33                                    Bx^�p  
�          @�z��a�@�\)@{A��B�k��a�@���@3�
AЏ\B��q                                    Bx^�~�  "          @�(��w
=@��@!G�A�Q�C.�w
=@��R@5AӅCz�                                    Bx^��d  "          @�G��z=q@�{@&ffAÅC��z=q@\)@9��A��Cs3                                    Bx^��
  �          @�z��y��@z=q@Mp�A�33C���y��@j=q@^�RBQ�C��                                    Bx^���  �          @���]p�@�Q�@FffA�C\�]p�@���@Y��BC��                                    Bx^��V  "          @�  �aG�@���@AG�A��B�z��aG�@��@VffA�Q�C8R                                    Bx^���  	�          @����S�
@��H@3�
A��HB�Ǯ�S�
@��
@J=qA�z�B�G�                                    Bx^�֢  T          @�=q�mp����?\An{Ce�{�mp����R?���A:�\Cf8R                                    Bx^��H  �          @�z���  ��(�?��A��\Ca����  ��  ?��HATz�Cb��                                    Bx^���  
�          @�ff����8Q�@#�
A��CQ#�����C�
@A��RCR��                                    Bx^��  
]          @�G������   @2�\A�
=CG��������@(Q�A�
=CI��                                    Bx^�:  
�          @����Q쿽p�@p�A�p�CA�)��Q��z�@A��HCCp�                                    Bx^��  �          @�ff����?Q�@AG�A��C,�����?
=@Dz�A��C.=q                                    Bx^�.�  �          @�Q����
@y��@hQ�Bz�C�
���
@g�@z=qB�
C
�3                                    Bx^�=,  �          @���s33@�33@QG�A�{C&f�s33@�33@e�B�\Cٚ                                    Bx^�K�  
�          @��H�j�H@���@S33A�\C���j�H@���@fffB�Ch�                                    Bx^�Zx  "          @��H���@^�R@h��B33C�
���@Mp�@xQ�B��C��                                    Bx^�i  T          @����H@Q�@o\)Bp�CJ=���H?���@x��B��C �                                    Bx^�w�  T          @����?�G�@"�\A�G�C*����?Q�@&ffA�Q�C,B�                                    Bx^��j  
�          @�
=���
?�  @p�A���C&33���
?��@z�A�=qC'�                                    Bx^��  �          @�����{?�\)@7�A̸RC!����{?�33@@  AָRC#Ǯ                                    Bx^���  W          @�33���H@@?\)A�p�CT{���H@
=@J=qA�ffCff                                    Bx^��\  Q          @��
��
=@\)@eBQ�CT{��
=@{@p��B
��C�H                                    Bx^��  T          @�33��ff@��@|��BG�C  ��ff@ff@��B��C�3                                    Bx^�Ϩ  
�          @��H��  ?�@��HB=qC  ��  ?�{@�
=B (�C!@                                     Bx^��N            @�ff��G�@Q�@���B�HCQ���G�@�
@��RB'Q�C��                                    Bx^���  �          @�z����@  @�(�Bz�C0����?�Q�@�G�B!p�CaH                                    Bx^���  �          @�{��\)@�@�z�B$ffC  ��\)?���@�G�B+��Cu�                                    Bx^�
@  
�          @�(����\@*�H@|��B�RC����\@�@�z�B�C�f                                    Bx^��  
�          @љ��@��@G�@�33B_ffC��@��?�\)@��Bh33C�R                                    Bx^�'�  T          @У��4z�@@���Bc�RCxR�4z�?�Q�@�G�Bm(�C(�                                    Bx^�62  "          @�
=�U�@C33@��\B5��C	���U�@,��@�G�B@�RC�                                    Bx^�D�  
�          @����H@
�H@��Bp��C
{��H?޸R@��B{\)C33                                    Bx^�S~  T          @�{�(Q�@@��Bh�
C}q�(Q�?�
=@��Br��CaH                                    Bx^�b$  �          @�ff��z�@Y��@\��B�C�)��z�@H��@l(�B�C�H                                    Bx^�p�  T          @�\)��p�@,(�@333A�p�C}q��p�@�R@@  A���Ch�                                    Bx^�p  �          @�Q���=q@Z=q?ǮA^ffCY���=q@Q�?�A�p�CT{                                    Bx^��  
�          @У�����@R�\?(��@��C:�����@N�R?h��A ��C�                                    Bx^���  "          @У�����@n�R����j�HCT{����@vff�����Ap�C��                                    Bx^��b  �          @љ���
=@^{�E����C8R��
=@l(��4z��υCz�                                    Bx^��  
Z          @�ff�vff@#�
@���B'�CY��vff@\)@��\B0��C                                    Bx^�Ȯ  "          @�33��Q�@aG�@dz�B�CxR��Q�@O\)@u�B33C�                                    Bx^��T  "          @�z���
=@}p�?�Q�A��
C  ��
=@s33@�RA�C#�                                    Bx^���  
�          @�{����@|��?��A/33Cc�����@u?�{AW
=C&f                                    Bx^���  T          @�
=���H@�33?c�
@��HC�����H@���?��HA ��CQ�                                    Bx^�F  �          @����  @��?�{A�C	�H��  @�
=?���AC�C
=q                                    Bx^��  
�          @޸R��G�@��
?�=qA33Cp���G�@���?�z�A;33C�                                    Bx^� �  W          @޸R��G�@�Q��G��k�C����G�@�Q�>k�?�z�C�=                                    Bx^�/8  �          @�ff���\@��H���H���\C����\@���8Q��  C��                                    Bx^�=�  
�          @�z�����@g
=��ff�|Q�C�H����@n�R�\�TQ�C�3                                    Bx^�L�  "          @��
��
=@;��S33��RC����
=@K��C�
��
=C�                                    Bx^�[*  �          @�=q��(�@8���HQ�����C�
��(�@G��9����z�C                                      Bx^�i�  "          @�G���@*�H�mp���\C���@<���_\)����C�=                                    Bx^�xv  �          @���\)@*�H�q�����C  ��\)@<���dz���C��                                    Bx^��  �          @�z���  @(������z�C���  @   �u����C�=                                    Bx^���  
�          @�p�����@L���o\)��RC������@^�R�^�R��ffC��                                    Bx^��h  
�          @�p���ff@
=q��(����Cff��ff@{�}p���HC��                                    Bx^��  "          @�p�����>�{���H�@p�C/�f����?@  ��G��>ffC+�                                    Bx^���  �          @�z����ff��
=�{��CaE������z��qCZ\)                                    Bx^��Z  "          @񙚿�z��'���=q#�Ci���z�����  
=Cc�\                                    Bx^��   "          @�33�{�4z���{�w=qCgǮ�{��\��z�W
Ca��                                    Bx^���  �          @��{�.�R�׮�y�
Cf�{�{�(���{�=C`�H                                    Bx^��L  �          @�\)��������
=���C]�{��Ϳ�z����
Q�CV(�                                    Bx^�
�  �          @��� �׿�\)��{.CP��� �׿Q�����CF.                                    Bx^��  T          @�33���33��\�=Cd#׿����R��
=C[                                    Bx^�(>  
�          @��\)�Ǯ��33�CV��\)��  ��ffaHCL�                                    Bx^�6�  
�          @��H��\��z����Hu�CS�)��\�Y����p�CHJ=                                    Bx^�E�  �          @�=q�������
=��Cg33��������\k�C[.                                    Bx^�T0  
�          @�\��������\C��;�׿����(�W
C}.                                    Bx^�b�  T          @������\��\Cd� �����p���
=\)C[u�                                    Bx^�q|  T          @�ff��ff����{�
Cjz��ff��\)���H�HCb+�                                    Bx^��"  T          @�(���{���׮B�Cgc׿�{�������C_��                                    Bx^���  T          @�ff�Q������ffCZ�f�Q쿰����p���CQ��                                    Bx^��n  
�          @�p��C33��p���=q�{{CQ�\�C33��
=��{z�CI!H                                    Bx^��  �          @�ff�y����p���\)�d�CH���y���s33��=q�jffCA��                                    Bx^���  "          @�  �e������\)�q��CH&f�e�E�����vC@�                                    Bx^��`  T          @�\)�XQ쿈���ۅ�z�\CE}q�XQ��\��p��~C<�\                                    Bx^��  �          @���fff�k����sz�CBY��fff�\��\)�v��C9�q                                    Bx^��  T          @�(��<��?Q��ڏ\�RC$^��<��?����׮z�C��                                    Bx^��R  
�          @���S�
>��R��  ��C.�{�S�
?\(���ff�{�
C%ff                                    Bx^��  T          @�Q���Q�=�����G��e��C2�=��Q�?�R��Q��dG�C+E                                    Bx^��  "          @��tz������  �nG�C5xR�tz�>�(��Ϯ�mp�C-��                                    Bx^�!D  �          @�Q�����>�p���\)�`�HC.������?aG����^{C(
                                    Bx^�/�  
�          @�Q��mp�=�\)�����q�C2���mp�?(���  �o��C*�                                    Bx^�>�  T          @�  ���
=�����Q��b��C2�����
?�R��\)�a  C+s3                                    Bx^�M6  T          @�����\=�����p��W{C2�����\?
=��z��U�\C,33                                    Bx^�[�  T          @陚��
=�:�H��ff�N�C=G���
=�������P33C7E                                    Bx^�j�  "          @�Q��u������R��CO�q�u���ff���\�Z�RCJ�                                    Bx^�y(  �          @�G��z=q���\��=q�]  CB��z=q����z��`C;Ǯ                                    Bx^���  
�          @�p�����?�Q���Q��Z(�C ff����?�
=���
�RCaH                                    Bx^��t  �          @����xQ�?�z���(��`  C �xQ�?�����XffC��                                    Bx^��  
�          @���c�
?   �θR�sC,\�c�
?�������o�C#�R                                    Bx^���  "          @�(���Q�?�ff��p��U�C&{��Q�?�ff����PG�C��                                    Bx^��f  
�          @陚�������
����W  C4:�����>����H�U��C-�H                                    Bx^��  
�          @�  ���>�=q���R�;�C0���?8Q���p��9�RC+��                                    Bx^�߲  "          @����p�@�
=�#�
��Q�C�3��p�@��R>��@XQ�C��                                    Bx^��X  �          @�R�]p�@��\��z��G�B�\�]p�@��p�����B�{                                    Bx^���  �          @�(���33?c�
���
�'��C*����33?�ff�����#�RC&O\                                    Bx^��  �          @����\@����'�HC����\@\)������Cٚ                                    Bx^�J  �          @����H@0����(����C����H@I�����
��C@                                     Bx^�(�  �          @�����@p������
CL�����@%�������C�q                                    Bx^�7�  
�          @�Q����\@�
���R��C�H���\@-p�������C�                                    Bx^�F<  �          @�R����@���e��33CaH����@,���W��׮C�                                    Bx^�T�  
�          @�(��ƸR@:�H�*�H��C���ƸR@I�������{C&f                                    Bx^�c�  
�          @�p��ȣ�@C�
�����
C���ȣ�@P�����C��                                    Bx^�r.  
�          @����z�@��C�
���C!#���z�@(��7
=��\)C�                                    Bx^���  �          @�R����?�Q��~{�=qC$p�����@��s�
����C!u�                                    Bx^��z  T          @�
=��  ?L���������C,n��  ?����R�
G�C(�                                    Bx^��   �          @�ff����?.{�z=q��{C-�{����?��\�u�����C*�q                                    Bx^���  �          @�  ��  @
=��  �(ffCǮ��  @2�\��Q��
=C�{                                    Bx^��l  
�          @�  ��=q?�Q����R�&z�C"c���=q@���G���\C8R                                    Bx^��  
�          @�R��(�<��
������C3ٚ��(�>�
=��Q���RC/��                                    Bx^�ظ  
(          @�ff��(�?�=q���R��\C)&f��(�?�  ���H��C%
                                    Bx^��^  X          @�  ���?aG������3
=C*n���?�{��p��.�RC%z�                                    Bx^��  	�          @�Q�����=�G������(�C2������?\)��  �'p�C.8R                                    Bx^��  
�          @�ff���H?u��ff��C*B����H?������H�ffC&
                                    Bx^�P  P          @�\)��p�@���\)���C ���p�@   �q����
C{                                    Bx^�!�  "          @�ff���\?������#z�C!k����\@�R���
=CJ=                                    Bx^�0�  
�          @�\)��ff?�33��=q� ��C#&f��ff@��z���C
=                                    Bx^�?B  �          @�\)��  ?�  ���
�8��C%�R��  ?޸R��\)�2�C �\                                    Bx^�M�  �          @�R��{?h�����R�'��C*z���{?�\)����#��C%�H                                    Bx^�\�  
�          @�ff����?�����
��C#�����@
�H�|(�� (�C�
                                    Bx^�k4  �          @�G�����>�
=����IC.������?u��  �F�C(��                                    Bx^�y�  "          @�����=L����z��N�C3Y�����?������M33C,��                                    Bx^���  "          @����@ �����H�\)C޸���@p���(��33C�H                                    Bx^��&  
�          @�33��{?�33��33�z�C"E��{@33�����G�C��                                    Bx^���  �          @������?�{��z��C'����?�ff�����C#
                                    Bx^��r  
�          @�p����?����33�G�C(����?˅��
=�G�C$�                                    Bx^��  �          @����ff>�{�����:p�C0=q��ff?^�R���R�7�HC*�=                                    Bx^�Ѿ  
�          @�p����ÿ
=�ʏ\�^�C;ٚ����<��
�˅�`�C3�q                                    Bx^��d  "          @�\)���
�#�
��Q��e�C<�
���
�#�
��G��g=qC4&f                                    Bx^��
  
Z          @����o\)�u�׮�p�
CBn�o\)���R����t�C8�                                    Bx^���  
Z          @�=q�q녿�����rC<���q�=�Q��ڏ\�t33C2�f                                    Bx^�V  T          @��H�Z�H��\��G��(�C<���Z�H>.{�ᙚ#�C1�                                    Bx^��  
�          @���`  ��=q����}�
C8k��`  >�
=��Q��}\)C-33                                    Bx^�)�  "          @�=q�hQ�&ff��z��wp�C>{�hQ�<���p��yQ�C3xR                                    Bx^�8H  "          @��\�Y���\)�������C=h��Y��>��ᙚ�=C1�                                    Bx^�F�  T          @�\)��=q=�Q���ff�G
=C2�f��=q?&ff����EG�C,B�                                    Bx^�U�  
�          @��R��  >�\)���\�%��C133��  ?G������#�C,=q                                    Bx^�d:  �          @��R�����\)��p��)ffC6ٚ���>k���p��)�C1��                                    Bx^�r�  &          @�{��>u��z��(�C1����?B�\���H�&z�C,k�                                    Bx^���  
�          @�\)��\)?8Q���(��0�C,� ��\)?�G������,�
C'�                                    Bx^��,  
�          @���  ?&ff��{�!C-����  ?�����H�(�C(�                                     Bx^���  
�          @�  �~{���R��{�b�\CH���~{�Tz�����i
=C?��                                    Bx^��x  
�          @�Q���
=������z��^{CC�{��
=�
=q��\)�bC;Q�                                    Bx^��  "          @�G���
=�Ǯ��G��Y�CHJ=��
=�h����p��`=qC@0�                                    Bx^���  &          @�(��
=>�  ��
=ǮC-�H�
=?��\���C��                                    Bx^��j  �          @�33��p�<��
���
�g�C3����p�?333�ҏ\�e{C*k�                                    Bx^��  T          @�33��(���=q�����C�
C6����(�>�{�����C��C0.                                    Bx^���  �          @�(�����?=p���33�X�RC*������?�33�Ǯ�SG�C"�{                                    Bx^�\  
V          @�33����>k���Q��H(�C1k�����?Y�����R�E��C*h�                                    Bx^�  T          @��\��=q?������D�RC-=q��=q?��H��=q�@z�C&��                                    Bx^�"�  
�          @����=q=�\)��{�PffC30���=q?8Q������Np�C+��                                    Bx^�1N  "          @�����H>.{���OC1�R���H?Q���(��MG�C*^�                                    Bx^�?�  T          @�����
������ff�3Q�C?�H���
�����G��7
=C9��                                    Bx^�N�  f          @�=q���\�k���z��:z�C6xR���\>�p���(��:(�C0�                                    Bx^�]@  �          @�=q��=q?�\��33�B�HC�)��=q@Q���(��933C�R                                    Bx^�k�  �          @�����?�p������?�HC#�����@���H�7��CQ�                                    Bx^�z�  �          @����G�?!G������E�C,�{��G�?�G������@�\C%�                                    Bx^��2  �          @��\��p�?�ff�����I
=C�R��p�@���G��>�
Cz�                                    Bx^���  �          @�G��\)@�z���\)�#��C���\)@�����  �(�C�{                                    Bx^��~  �          @���c�
@�ff�����p�B�z��c�
@�{��
=��HB�p�                                    Bx^��$  �          @�����33@�{��p���Ch���33@������(�CxR                                    Bx^���  �          @����
@�����\�%ffB�ff���
@�\)��ff�=qB�\)                                    Bx^��p  �          @�z���{������p��6
=CA���{��������:Q�C:��                                    Bx^��  �          @�p����R������p��4��CBǮ���R�E���G��9�RC<c�                                    Bx^��  �          @�ff���H��=q���R�*��CF����H���R��z��1�
CA)                                    Bx^��b  �          @�{��33��33�����/  CB����33�O\)��p��4(�C<��                                    Bx^�  �          @�����Q쿚�H�����)�C@h���Q�!G���Q��-�C:��                                    Bx^��  �          @������G������.  CA�����(����(��2�C:�3                                    Bx^�*T  �          @�������E���Q��.  C<�����8Q�����033C5ٚ                                    Bx^�8�  �          @�z���=q����p�� ��C8����=q=�����{�!z�C3\                                    Bx^�G�  �          @���z��R��Q��G�C9�q��z὏\)�������C4��                                    Bx^�VF  �          @�(���33�У�����z�CDB���33��=q�����"�C>�                                    Bx^�d�  �          @�(���33��p���{�!��CB��33�fff���\�'G�C=#�                                    Bx^�s�  �          @�p���zῧ������#��CA)��z�:�H��(��(Q�C;Y�                                    Bx^��8  �          @�p���33��33��(��=qCF���33��=q��=q�%�\CAQ�                                    Bx^���  �          @�p����\���
�p�CB�����p����Q��${C=n                                    Bx^���  �          @�����׿����z��=qC>(����׾���\)�"�C8��                                    Bx^��*  �          @�{��p��333�����G�C:��p��\)��33�33C5Q�                                    Bx^���  �          @����Ϳ�ff��ff�!��CA  ���Ϳ8Q�����&�\C;5�                                    Bx^��v  �          @���z�������RC9\��z�=u��z��C3u�                                    Bx^��  �          @���(�?(�����\�C-����(�?�(����R�z�C(E                                    Bx^���  �          @�\)��?^�R��G��G�C,���?�33����\)C'8R                                    Bx^��h  �          @�\)���R?�\)���H�Q�C$�����R@	������

=C &f                                    Bx^�  �          @�����?�{��p��C%&f���@	����{�
��C k�                                    Bx^��  �          @�����=q?�����
=��\C)s3��=q?�Q������G�C$s3                                    Bx^�#Z  �          @������?��R��
=�Q�C(s3���?�ff������\C#xR                                    Bx^�2   �          @������
?
=�\)���
C.�H���
?����x�����HC*�=                                    Bx^�@�  �          @�����ff>����p���HC2�{��ff?8Q����
��HC-^�                                    Bx^�OL  �          @�=q���>�z���ff�
��C1^����?W
=��(���C,��                                    Bx^�]�  �          @�����(�>�z���p��
��C1h���(�?Tz���33�(�C,��                                    Bx^�l�  �          @����ȣ׿��������C@� �ȣ׿W
=��  �C;��                                    Bx^�{>  �          @�=q��p���Q���33����CB�=��p��������  C>\)                                    Bx^���  �          @��\�Ϯ��G���
=��C<���Ϯ�������(�C8.                                    Bx^���  �          @���ƸR���
�������C4�R�ƸR?��������C/5�                                    Bx^��0  �          @��
�ƸR?aG�����\)C+��ƸR?�(���33�  C&��                                    Bx^���  �          @�33����?�  ��z����C*�)����?�{��\)��C%�                                    Bx^��|  �          @��\���R?�������!ffC$  ���R@�������
=Cff                                    Bx^��"  �          @�33����?�������4�\C'k�����?�33����,�C �)                                    Bx^���  �          @�z����R?�  ����3(�C')���R?�������*��C aH                                    Bx^��n  �          @�{����?�G����H� �C#!H����@����=q�=qC�\                                    Bx^��  �          @����@���=q�*{C� ��@,����Q��G�C��                                    Bx^��  �          @�33����@���z����C!Q�����@(���u���RC5�                                    Bx^�`  T          @�=q��33@  ����z�C"k���33@!녿�(��j�HC h�                                    Bx^�+  �          @�  �׮@9�������\C�q�׮@L(����t  C��                                    Bx^�9�  �          @�����@U�QG���Q�C�\���@n�R�3�
��z�C�                                    Bx^�HR  �          @�����@Vff���H��Q�Cs3��@w
=�g
=��  C�\                                    Bx^�V�  �          @�G����
@x����33���C{���
@����c33�؏\C�3                                    Bx^�e�  T          @�\)��{@=p��Z�H���Cz���{@XQ��@  ��=qC\)                                    Bx^�tD  �          @����@j=q�Q����\C�)���@|�Ϳ�\)�b{C�R                                    Bx^���  �          @������@x����
�uG�C�q���@�(��\�4(�C33                                    Bx^���  �          @�����p�@��\����_�C�3��p�@�G����
��C��                                    Bx^��6  �          @�����@�\)������C���@��\�Ǯ�:=qC5�                                    Bx^���  �          @���G�@�{�n{��ffC�{��G�@��׾aG���\)Ch�                                    Bx^���  �          @����
=@'
=�����0\)C�R��
=@QG����
�G�C�)                                    Bx^��(  �          @�����p�@�녿�Q��HQ�C���p�@���h����  C                                    Bx^���  �          @�ff��
=@�  �}p����
C 0���
=@\�#�
��
=B���                                    Bx^��t  �          @����  @�
=�������C(���  @�Q���H�K33C�H                                    Bx^��  �          @����33@��R�0����  Ck���33@�G��G��pQ�C�f                                    Bx^��  �          @�����=q@�Q��%��  C(���=q@�녿��W33C��                                    Bx^�f  �          @������
@�33�.{���HCL����
@�p������j{C��                                    Bx^�$  �          @��R��{@�p��Q���G�C
��{@��\�#�
���C�\                                    Bx^�2�  �          @�Q����
@��A����RC=q���
@��������CB�                                    Bx^�AX  �          @�Q���G�@]p�������RC#���G�@�����
�
�CQ�                                    Bx^�O�  �          @����Q�@z�H��  �(�C���Q�@���������C�q                                    Bx^�^�  �          @�ff����@w
=��  �"{C
������@����������C(�                                    Bx^�mJ  �          @��R����@:�H��{�5
=C.����@j�H��
=�!�
C                                    Bx^�{�  �          @�z�����@U���
�2�
C�q����@�=q���\�Q�CaH                                    Bx^���  �          @�z���
=@a����\�&C����
=@�
=�����(�C��                                    Bx^��<  �          @��
�u�@�p����
�
=C�R�u�@�=q���Q�B�33                                    Bx^���  �          @�p��j=q@��H������C �
�j=q@�  ��ff���B���                                    Bx^���  �          @�  ��
=@}p���{�
=C  ��
=@�33��=q��G�C�                                     Bx^��.  T          @��R��33@�����
=�{C��33@��H�e����C
�                                    Bx^���  �          @�\��=q@�  �i������C���=q@���?\)��(�C�3                                    Bx^��z  T          @��
����@�p������
=C������@���\����\)C5�                                    Bx^��   �          @�����ff@��
�b�\��G�CY���ff@��\�0  ���B��)                                    Bx^���  �          @�=q�AG�@��\�mp���B����AG�@�=q�3�
���RB�#�                                    Bx^�l  �          @��\)@�33�b�\��B�W
�\)@���%��=qB݅                                    Bx^�  �          @��H���@�=q���
�"��B�����@Ǯ�~�R� �
B��                                    Bx^�+�  �          @�=��
@�p������;B�=q=��
@�{��ff�Q�B�u�                                    Bx^�:^  �          @�Q��@�ff����*ffB�B���@�����
=�	�B�                                      Bx^�I  �          @�׿�{@�����ff�G�B�.��{@�(��a���B�W
                                    Bx^�W�  �          @�
=�g
=@���C�
��z�B�aH�g
=@�z��
=q����B���                                    Bx^�fP  �          @���.�R@�z��[��ޣ�B�R�.�R@��H�\)���RB�\                                    Bx^�t�  �          @�(�����@x���xQ����HC������@�{�O\)��\)C�                                    Bx^���  �          @�Q����@=p����
� �C8R���@c33�g�����C                                    Bx^��B  �          @�����{@1���{�G�C����{@X���mp����CE                                    Bx^���  �          @���\@����z���Cu��\@@���n{��(�C�f                                    Bx^���  �          @�ff�Ǯ@���r�\��ffC��Ǯ@<���W���Q�C�3                                    Bx^��4  �          @�
=�ƸR?У���Q��=qC%E�ƸR@��}p�����C�H                                    Bx^���  �          @�\)�ʏ\?�G�������C&�=�ʏ\@���w
=��=qC!G�                                    Bx^�ۀ  �          @�\)��Q�@
�H�|����Q�C �)��Q�@1G��c�
��C+�                                    Bx^��&  �          @�Q���{?��H�w
=��RC#���{@"�\�`�����HC}q                                    Bx^���  
�          @����Ϯ@Q��n{��RC!�)�Ϯ@,(��U��\)C�                                     Bx^�r  �          @���ָR?�
=�h�����HC*��ָR?޸R�Z=q��\)C%}q                                    Bx^�  �          @�  �߮>��Tz���33C2��߮?(���P�����C.��                                    Bx^�$�  �          @�Q��޸R>����XQ����C1���޸R?W
=�R�\���C-#�                                    Bx^�3d  �          @�Q����ÿ333�j�H��(�C9�����ý����n�R��\C4��                                    Bx^�B
  �          @���׮�L���j=q���HC:�=�׮�L���o\)��ffC5��                                    Bx^�P�  �          @�
=���H?(���<(���\)C.u����H?����2�\���C*��                                    Bx^�_V  �          @�����@��   �p(�C#�)��@�����C�C!��                                    Bx^�m�  �          @�
=��>�ff�$z���{C0h���?^�R�p���G�C-+�                                    Bx^�|�  �          @�{����?��������C/������?u�������C,��                                    Bx^��H  �          @��R��\>����33���C1h���\?333�p����
C.��                                    Bx^���  �          @�\)��녿���{����C8E��녾��!G���Q�C5�                                    Bx^���  �          @�\)��׾�Q��%���\C6޸���=�\)�'
=��{C3u�                                    Bx^��:  �          @�����  >�z��.�R����C1� ��  ?=p��)������C.5�                                    Bx^���  �          @�G�����G��p���=qC7ff����\)�  ���HC4��                                    Bx^�Ԇ  �          @�G���
=�u�
�H����C5����
=>�����(�C3�                                    Bx^��,  �          @�����p��G��
�H��{C9�q��p������G���=qC7�                                    Bx^���  �          @�����  ����\�r�RC8O\��  �W
=�
=�z�RC5�H                                    Bx^� x  T          @�����R�5���r{C9n��R��Q����}�C6��                                    Bx^�  �          @�ff���H�����{�`z�C>#����H�z�H��
�x��C;�{                                    Bx^��  �          @�ff��R�Q녿ٙ��Lz�C:=q��R��\�����Z�\C7�                                    Bx^�,j  �          @���Q�=#�
�����=G�C3�3��Q�>��R����9��C1�f                                    Bx^�;  �          @�ff��?333�����\C.�3��?h�ÿ�33�	��C-�                                    Bx^�I�  T          @�ff���H>Ǯ���\��
C1
���H?����
=���C/z�                                    Bx^�X\  �          @�ff��\>�׿�p���HC0z���\?+������
=C.�                                    Bx^�g  �          @�p����H>�(���G���=qC0�����H?���k���z�C/u�                                    Bx^�u�  �          @�{��33?5�p����Q�C.�H��33?^�R�L����
=C-}q                                    Bx^��N  �          @�Q���Q�?��Ϳ�����ffC'�R��Q�?�G��G����C&�=                                    Bx^���  T          @����>�33�z���33C1aH��>�G���\�u�C0��                                    Bx^���  �          @�����\?L�Ϳ
=��C.���\?c�
��G��S33C-Q�                                    Bx^��@  �          @�����R?�녾�����C'����R?��#�
��\)C'h�                                    Bx^���  �          @�z���p�?�p����_\)C&�H��p�?�������\)C&z�                                    Bx^�͌  �          @�33����?�
=��Q��/\)C'.����?�(��u��(�C&�f                                    Bx^��2  �          @�\��?��H��\)��\C(����?���>B�\?�(�C(�R                                    Bx^���  �          @����?��׽�G��Q�C)}q��?�\)>\)?���C)�                                    Bx^��~  �          @�  ��
=@논#�
����C$G���
=@   >�33@,��C$��                                    Bx^�$  �          @�  ��ff?�(�>Ǯ@>{C$���ff?�\)?=p�@�p�C%p�                                    Bx^��  �          @�\)���@�\?G�@�  C"����@
=?�
=A�C#\)                                    Bx^�%p  �          @�R��ff@'
=?�@��\CaH��ff@{?�  @�ffC ff                                    Bx^�4  �          @������@E�?k�@�{C&f���@7
=?�(�A7�
C�q                                    Bx^�B�  �          @�p����@)�����
�8Q�C  ���@'
=>�@eCL�                                    Bx^�Qb  �          @�R����@+��8Q쿰��C�=����@*=q>��
@p�C��                                    Bx^�`  T          @�Q���\)?�{�O\)�ƸRC%�{��\)?�p���ff�^{C$��                                    Bx^�n�  �          @�����H@ff���R��
C#�=���H@�\�Tz���=qC"�                                    Bx^�}T  T          @�\)����?˅�����\)C'������?��u���C'�                                    Bx^���  �          @�����@(���(��R�\C#\��@�R���
��C"                                    Bx^���  �          @��陚?������"�\C%�
�陚?�\)<��
>�C%��                                    Bx^��F  �          @��H��=q@\)��z����C ����=q@*�H�+����C\)                                    Bx^���  �          @�=q��@��c�
�ٙ�C!�)��@p���G��Tz�C �3                                    Bx^�ƒ  �          @�33��
=@8Q�c�
��Q�C����
=@@  �����   C��                                    Bx^��8  �          @�\���H@l�;��`��C�����H@n{>�  ?��C�=                                    Bx^���  �          @����
=@J=q��Q�+�C����
=@G�?�@w
=C�f                                    Bx^��  �          @������@AG�>�=q@   C�q����@:�H?W
=@ə�C}q                                    Bx^�*  �          @��
����@Z�H>�p�@2�\C=q����@Q�?��\@�
=C&f                                    Bx^��  �          @��H��z�@fff?\)@��RC����z�@Z�H?��RA�RC�R                                    Bx^�v  �          @���У�@mp�?c�
@�\)CW
�У�@^{?˅AB{C�R                                    Bx^�-  �          @����G�@Dz�?O\)@�ffC�H��G�@7
=?��A+33C(�                                    Bx^�;�  T          @�\)��G�@aG�?O\)@ƸRC�3��G�@S33?�p�A6�RC@                                     Bx^�Jh  T          @����  @:=q?aG�@��
C����  @+�?�Q�A3
=CY�                                    Bx^�Y  �          @���G�@1�?�{A	G�C�R��G�@ ��?У�AK33C�3                                    Bx^�g�  �          @����\)@\��@�A33C���\)@>�R@*�HA�\)Cs3                                    Bx^�vZ  �          @���
=@H��?L��@�(�C���
=@:�H?�33A-G�Cz�                                    Bx^��   �          @���
=@J=q?W
=@�ffC�\��
=@;�?���A2�HCn                                    Bx^���  �          @�\)�ٙ�@5?�33Ap�CT{�ٙ�@#�
?�Q�AQG�Cff                                    Bx^��L  �          @���@&ff@�A|z�C����@��@ ��A�33C"5�                                    Bx^���  �          @�  �أ�@B�\?
=q@�C���أ�@7�?��A��C�                                    Bx^���  �          @�  ��ff@r�\@{A�Q�CxR��ff@N{@K�AɅC��                                    Bx^��>  �          @���z�@J�H@(�A�Q�C����z�@*�H@1�A��CQ�                                    Bx^���  �          @�Q���\)@l��@E�A¸RC+���\)@@  @qG�A��Cff                                    Bx^��  �          @�Q���  @n�R@   A�  C���  @I��@Mp�A���CO\                                    Bx^��0  �          @�=q��=q@���@7
=A��\C�R��=q@XQ�@g�A���C�R                                    Bx^��  �          @�\�\@z�H@
=A�Q�C+��\@Z=q@7�A�33C��                                    Bx^�|  �          @�=q���@��@Q�A��C�����@h��@<(�A��CaH                                    Bx^�&"  �          @����
=@���@5A�ffC���
=@XQ�@g
=A�Q�Ch�                                    Bx^�4�  �          @�=q��Q�@~{@S�
A�C5���Q�@Mp�@��B��C�{                                    Bx^�Cn  �          @�=q��\)@�  @:=qA�
=C���\)@S�
@k�A�\C�R                                    Bx^�R  �          @����@�\)@(��A��C=q���@e@]p�A�Q�C�H                                    Bx^�`�  �          @�������@���@>{A�{C�����@\(�@qG�A�RC{                                    Bx^�o`  �          @������@�Q�@Mp�A�G�C!H����@P  @~�RA�=qC��                                    Bx^�~  �          @��H��=q@���@]p�A�p�C
��=q@U�@�Q�B��C                                      Bx^���  �          @�����33@�z�@{�A���C�\��33@N{@��RB�HCxR                                    Bx^��R  �          @���ڏ\@O\)���
��  C���ڏ\@XQ쾣�
�=qC��                                    Bx^���  �          @�\)�Ӆ@333�7
=���C{�Ӆ@U�����ffC5�                                    Bx^���  T          @���@N{���g�C
��@P  >W
=?˅C��                                    Bx^��D  T          @�
=�޸R@O\)�
=����C��޸R@R�\=�G�?Y��C�                                    Bx^���  �          @�  ��{@8�þ��XQ�C
��{@:�H>.{?��\C�H                                    Bx^��  �          @�{�أ�@aG�?W
=@�Q�C���أ�@P  ?���A=�CQ�                                    Bx^��6  �          @�����@e�?�Q�A�RC�����@O\)?�
=AiC
=                                    Bx^��  �          @�ff�ʏ\@��?��@��C&f�ʏ\@z=q?���Ak
=CB�                                    Bx^��  �          @��ȣ�@���?�p�A
=C�ȣ�@y��@�A�ffC�                                    Bx^�(  �          @�33��ff@xQ�?�  @�G�C���ff@dz�?���A]��C�q                                    Bx^�-�  �          @��У�@p  ?p��@��C)�У�@\��?޸RATQ�C)                                    Bx^�<t  �          @�����
@�33?�33A	�C:����
@p  @G�AtQ�C�                                    Bx^�K  �          @����G�@�33?��RAvffC)��G�@s33@9��A��C��                                    Bx^�Y�  �          @�Q����H@�G�@@��A�ffC
:����H@qG�@|(�A��HCu�                                    Bx^�hf  T          @�ff����@R�\@θRB{��B������?��
@���B�=qB�Q�                                    Bx^�w  �          @�R�Tz�@�  @�{Bb=qBˀ �Tz�@!�@�p�B��fB�k�                                    Bx^���  �          @�\)����@��@ÅB[�\B��Ϳ���@,��@�z�B�G�B�33                                    Bx^��X  �          @�Q�8Q�@��H@�Q�Bc(�B�(��8Q�@%@�Q�B�Q�B�#�                                    Bx^���  �          @���<#�
@�G�@��HBe�B�Ǯ<#�
@!G�@�\B���B���                                    Bx^���  �          @�Q�>�ff@P��@�B
=B�L�>�ff?�
=@�B�  B�                                    Bx^��J  �          @���?p��@B�\@�G�B�ǮB��{?p��?�
=@��B��\Bb                                    Bx^���  �          @��?#�
@qG�@�z�Blp�B�?#�
@�R@�=qB��RB�                                    Bx^�ݖ  �          @�@�\@n�R@�(�B]
=Bu(�@�\@\)@�=qB�B>�R                                    Bx^��<  �          @��@\)@��R@�G�BJG�Bw��@\)@1�@�33BvffBL�                                    Bx^���  �          @�ff�#�
@���@���BU�B�#׿#�
@5@��
B�z�B͏\                                    Bx^�	�            @�\)�A�@�
=@�z�B$G�B�Q��A�@^�R@��HBNC                                    Bx^�.  T          @��S33@�z�@�33B�B�\�S33@mp�@�33BA�RC�3                                    Bx^�&�  T          @�  �AG�@��
@��B �B�q�AG�@hQ�@���BKffC��                                    Bx^�5z  �          @���*=q@���@���B3�HB���*=q@Mp�@�B_�
C��                                    Bx^�D   �          @�Q��K�@��@���B�B��
�K�@e�@�Q�BI(�C��                                    Bx^�R�  �          @�=q��@�p�@���BM�B�p���@;�@�Q�B}�B�ff                                    Bx^�al  �          @񙚿�p�@�G�@�33BL�RB�G���p�@C�
@�  B�HB���                                    Bx^�p  �          @�=q���@��H@��B?ffB�G����@X��@ҏ\Bq�B�                                    Bx^�~�  �          @�G���33@��@�\)B:�B����33@X��@�ffBkB�                                    Bx^��^  �          @�Q��
=q@���@��HB@�RB�
=�
=q@E�@�  Bo�
B�\                                    Bx^��  �          @�G��   @�{@��\B4\)B�\�   @S33@�G�Bb�B�.                                    Bx^���  �          @�  �A�@���@�G�B
=B�\�A�@hQ�@��BK�\C�                                    Bx^��P  �          @�\�y��@�ff@���Bp�C 5��y��@u�@�ffB.z�C�                                    Bx^���  �          @�\�|(�@�\)@��\B�HC ^��|(�@w�@���B+��C��                                    Bx^�֜  
�          @��H��Q�@���@p��A�CG���Q�@~�R@��HB�HC�f                                    Bx^��B  �          @��
���\@�Q�@A�A��
C
=���\@�z�@�ffB{C@                                     Bx^���  �          @�33�Vff@�33@�Q�Bp�B�8R�Vff@l(�@���B;{C:�                                    Bx^��  �          @�  ���R@w
=@��B_p�Bר����R@�
@׮B�u�B�\                                    Bx^�4  �          @����33@q�@�
=Ba��B���33@�R@�ffB���B�p�                                    Bx^��  �          @�p���@p��@�  Bc
=B֊=��@(�@�\)B�z�B�{                                    Bx^�.�  �          @��H��=q@���@��BX\)B�
=��=q@   @�G�B�ǮB�{                                    Bx^�=&  �          @�\�^�R@�=q@�ffBWp�B���^�R@#�
@У�B��B�L�                                    Bx^�K�  �          @��
�Y��@o\)@�ffB`G�B͞��Y��@\)@�{B�B�Bݣ�                                    Bx^�Zr  �          @��Ϳ   @�
=@��RBD��B��R�   @C�
@�z�B|Bƀ                                     Bx^�i  �          @��333@���@�ffB]�HBǮ�333@��@�  B��B��H                                    Bx^�w�  �          @�=q��ff@p  @�(�BdffB�W
��ff@�@ۅB��B�\                                    Bx^��d  T          @�z῾�R@g�@�  Bg�
B�׿��R?��H@�ffB���B�.                                    Bx^��
  T          @�p���=q@j=q@��
B\��B�=q��=q@@�33B�L�C@                                     Bx^���  �          @ڏ\��@`  @��B]ffB��f��@   @ə�B��C��                                    Bx^��V  �          @�\)���
@r�\@�p�BX��B�  ���
@��@�{B�  B�8R                                    Bx^���  �          @�녿��\@q�@��BX�B�(����\@�\@�Q�B���B�\                                    Bx^�Ϣ  �          @�(����H@���@�Q�BHffB��)���H@,��@�(�B}�\B��f                                    Bx^��H  �          @�Q쿘Q�@n{@�z�BM��B�p���Q�@�@�p�B���B�8R                                    Bx^���  �          @�(��Ǯ@l(�@�ffBL
=B��
�Ǯ@�@�
=B��B��3                                    Bx^���  �          @�\��@�
=@�ffB2\)B�����@E�@�p�BdQ�B���                                    Bx^�
:  �          @�Q��z�@���@��RBC33B�aH�z�@1�@��
Bu�B�L�                                    Bx^��  �          @�z��(�@���@��BB�\B�ff��(�@1�@ȣ�BuB��{                                    Bx^�'�  �          @�  ��@�Q�@�z�B?��B�G���@0  @ə�Bp�C@                                     Bx^�6,  �          @�(����@�G�@��HBB�B�p����@"�\@�ffBq\)C�                                    Bx^�D�  �          @��'�@~�R@���B?ffB����'�@   @ÅBmG�Cc�                                    Bx^�Sx  
�          @�
=�<(�@c�
@��
B9��C�{�<(�@(�@��
BcQ�CL�                                    Bx^�b  �          @�����@e@�p�BL��B�\��@��@�p�B|G�C�R                                    Bx^�p�  �          @�\)��
=@z�H@��BQffB՞���
=@�H@�z�B���B�(�                                    Bx^�j  �          @�  ���@��@��B3��B�=q���@J=q@�=qBkp�B���                                    Bx^��  �          @���@�  @�(�B=z�B�=q��@L(�@�Bv
=B��                                    Bx^���  �          @�\)���@���@���BI\)Bݨ����@-p�@�
=B��B�=q                                    Bx^��\  �          @�(�����@�z�@�Q�B4{B�W
����@Z=q@ÅBnp�B׮                                    Bx^��  �          @��ÿ���@���@��B>{B�𤿐��@R�\@�=qBxp�B���                                    Bx^�Ȩ  T          @�zῈ��@���@��B<��B��Ὲ��@Z=q@�{Bwp�B��)                                    Bx^��N  �          @�(��
=@��@��B(�B����
=@���@\BT�
B�#�                                    Bx^���  �          @��
��R@�ff@�Q�BffB���R@��@�33BI33B�R                                    Bx^���  �          @����'
=@�Q�@��\B��B�q�'
=@��@�(�BIffB��                                    Bx^�@  �          @�{����@�  @��RBBծ����@�33@�z�BG�
B�\                                    Bx^��  T          @�녿��
@�p�@�\)Bz�Bʔ{���
@�=q@���B?��B�ff                                    Bx^� �  �          @�(�����@�ff@�(�Bp�B�
=����@�  @ÅBLp�B�{                                    Bx^�/2  �          @�zῨ��@��H@��HBQ�B�aH����@��\@���BSffB�\                                    Bx^�=�  �          @�zῦff@�  @��B0�BΔ{��ff@u�@أ�BlffB�p�                                    Bx^�L~  �          @��.{@���@�(�B+33B�L;.{@���@׮BiffB���                                    Bx^�[$  �          A Q�=u@�G�@���B8��B��
=u@q�@�G�Bw  B�L�                                    Bx^�i�  �          @�ff�u@�(�@��RB%��B��u@�Q�@�33Bdp�B��\                                    Bx^�xp  �          @�ff=�@�(�@�33B(�B��=�@�ff@�\)BC�HB�aH                                    Bx^��  �          A\)���@�33@j�HA�=qB����@\@���B(��B��)                                    Bx^���  �          A z�?E�@��
@���B��B�G�?E�@��\@ʏ\BQ=qB���                                    Bx^��b  �          @�=q?#�
@�p�@hQ�A�33B�\?#�
@�@���B0z�B��{                                    Bx^��  �          @�ff��ff@�
=���H�n�RB�\��ff@�=q?���A?\)B�aH                                    Bx^���  �          @�  �z�@����!G����B�Ǯ�z�@�33�+����\Bՙ�                                    Bx^��T  T          @����@��
�
=��33B���@����:�H��z�B���                                    Bx^���  �          A�
?�\)>�Q�A�B�{Ai��?�\)���@��B�\C��                                    Bx^���  �          A33?�p����RA z�B���C��?�p��i��@陚Bq��C�C�                                    Bx^��F  �          A�@�>ǮA33B�z�A,Q�@녿��HA�B�{C�                                      Bx^�
�  �          A33?���?�ffA�RB��A�z�?��Ϳ���A{B���C�.                                    Bx^��  �          A
=?��?��A	p�B�aHBJz�?���z�Az�B��RC���                                    Bx^�(8  �          A(�?�33?ٙ�A�B���BI��?�33�L��AB�\)C�1�                                    Bx^�6�  �          A�?�p�?У�A33B���B>��?�p��\(�A��B��C��                                     Bx^�E�  �          A�H?�{?�33A	�B���BFQ�?�{�z�A(�B��
C�'�                                    Bx^�T*  �          A=q?Ǯ?�p�A	�B���B?�
?Ǯ�=p�A\)B��\C�@                                     Bx^�b�  �          A�?�{?˅A	G�B�u�B3(�?�{�aG�A
�HB�.C���                                    Bx^�qv  �          A��?޸R?�(�A�
B���B2�?޸R�=p�A
{B�L�C��                                     Bx^��  �          A��@�
?��A�B�z�A�(�@�
���A�RB�
=C��)                                    Bx^���  �          AQ�@�>��
A z�B�W
@��R@���(�@���B�8RC��                                    Bx^��h  �          A\)@7
=���R@��B��C�5�@7
=�i��@�=qBb��C�
=                                    Bx^��  �          A33@5�У�@���B�aHC��@5�q�@�  B_�C�w
                                    Bx^���  �          A{@#�
��p�@��HB���C��@#�
�j=q@��HBgG�C�|)                                    Bx^��Z  �          A�@���A   B��HC�  @�a�@陚Br��C�Q�                                    Bx^��   �          A
�R@$z�?(�A  B��=ATQ�@$z���
AG�B��C��q                                    Bx^��  �          A	p�@�R?
=A33B�z�AW
=@�R���A Q�B�Q�C��                                    Bx^��L  �          A
�R@5�?�  A (�B�p�A�\)@5��^�RAB��HC�t{                                    Bx^��  �          A�
?�p�?Tz�A
=B���A�\)?�p�����AG�B�G�C��                                    Bx^��  �          A=q?��H>�p�AffB���AD  ?��H�   @��B�8RC�7
                                    Bx^�!>  �          A{?��H?��RA ��B�ǮB�?��H���A ��B��{C��                                    Bx^�/�  �          A�H?��
?�(�A{B�33B	�R?��
��Q�A=qB�u�C�#�                                    Bx^�>�  T          A��?��?�z�A�RB��=BLG�?�����A{B��)C�1�                                    Bx^�M0  �          A��?��
@�A�RB�p�BTG�?��
��G�A�\B���C��{                                    Bx^�[�  �          A��?���@�\A{B��Bf?����B�\A
=B�\)C�Z�                                    Bx^�j|  �          A��?�z�@{A ��B�Bqz�?�z�<#�
A�RB�>�33                                    Bx^�y"  �          AQ�?��@�A ��B�W
Bn(�?�녽���AffB��C��                                    Bx^���  �          A��?�=q@(�Ap�B�\Bu��?�=q�L��A33B��C��)                                    Bx^��n  �          A��?�Q�?У�A\)B��B/��?�Q�^�RAG�B�G�C�l�                                    Bx^��  �          A	G�@   ?0��A��B�k�A��@   ��ffA=qB�k�C��                                    Bx^���  �          AQ�@
=q��G�A�B��=C���@
=q�"�\@�=qB��
C�/\                                    Bx^��`  �          A�@ff=�G�A
=B�8R@@��@ff��@��B�� C�                                    Bx^��  �          A�
?�{?�A�\B�Bff?�{����A\)B��3C��)                                    Bx^�߬  �          A  ?��?��
A=qB��
Bp�?�׿xQ�A�B�(�C�]q                                    Bx^��R  
�          A�
?�?\A=qB�W
Bp�?��z�HA�B��\C���                                    Bx^���  �          A�@33?��RA�B�(�B�@33��  AffB��3C���                                    Bx^��  �          A�H?�33?���A=qB�ǮB$�?�33���A33B�\C���                                    Bx^�D  �          A�\?��?�=qAp�B�.B(�?����A�B�z�C�u�                                    Bx^�(�  �          A@�R>�@�
=B���A'�@�R���H@�\)B�u�C��
                                    Bx^�7�  �          A�\?�{?��A�HB�z�B  ?�{���HA�B���C���                                    Bx^�F6  
�          A�R>�33?���@��B�\)B�=q>�33�W
=@�p�B��)C�Ff                                    Bx^�T�  T          @�p��I��@�(�@C33A�RB�ff�I��@_\)@��B*=qC\                                    Bx^�c�  �          @�33�z�H@���@ffA��C�f�z�H@fff@g�B�
C	xR                                    Bx^�r(  T          @���
=@���{�-p�C�H��
=@���?�
=A
=C	ff                                    Bx^���  T          @����{@�p����R�D��C}q��{@�������HC�{                                    Bx^��t  �          @�33��ff@�Q��=q�F�RC��ff@�����\)�
�HC#�                                    Bx^��  T          @����Q�@�녿����ffC���Q�@�\)=���?E�C�                                    Bx^���  �          @�  ��z�@�
=�aG���\)C���z�@���>�@`��C)                                    Bx^��f  �          @���p�@��\��{�j�\C����p�@���Q��1�C	��                                    Bx^��  �          @陚����@����Q��VffC�\����@��R��  ��Q�C��                                    Bx^�ز  T          @����33@��Ϳ���O�C޸��33@�{�L�Ϳ���C�                                    Bx^��X  �          @����=q@��
��ff�d  C޸��=q@��R��Q��2�\C                                    Bx^���  T          @����
=@�(������yG�CL���
=@��׿   �z�HC
�f                                    Bx^��  T          @���z�@�Q��p��]�C���z�@�=q�u��z�C
0�                                    Bx^�J  T          @�p���@�
=�����1��C����@�=��
?&ffCE                                    Bx^�!�  T          @����
@��
��z��uG�C� ���
@�Q����{C=q                                    Bx^�0�  �          @��H���R@���z���  C
G����R@���?z�H@�z�C
��                                    Bx^�?<  �          @��H���@\)����|Q�CJ=���@�(��
=q��ffC�3                                    Bx^�M�  
�          @���z�@!���
=�(�C���z�@}p��z�H��G�C                                    Bx^�\�  �          @���@5���(��G�Cc���@���QG����C�                                    Bx^�k.  �          @�\���@:�H��G���\Cu����@��
�I�����CW
                                    Bx^�y�  T          @陚��G�@L(���
=�Q�CY���G�@�{�N�R�ң�C	0�                                    Bx^��z  �          @�����@Y����G��=qC�����@��H�?\)���C^�                                    Bx^��   �          @����  @9�����\��
C
��  @�G��<�����
Cff                                    Bx^���  �          @�������@:�H���H���C�����@����L(���  C��                                    Bx^��l  �          @�\��33@:=q������C�q��33@�  �_\)��C
Ǯ                                    Bx^��  �          @�33��\)@3�
���
��HC�
��\)@�(��_\)��z�CJ=                                    Bx^�Ѹ  �          @��H���R@p���G��.  Cs3���R@|(��\)��C
                                    Bx^��^  �          @��
��  @  �����
C����  @g��l(����CaH                                    Bx^��  �          @�ff���R@Q���\)��RCxR���R@l(��^�R�ޏ\C��                                    Bx^���  �          @�\)��=q@\)�����*��C���=q@l���z�H�p�C�                                     Bx^�P  �          @���Q�@{��z��.Q�CQ���Q�@z=q�u���C�                                    Bx^��  �          @�{��{@"�\����;��C����{@��H��z���C��                                    Bx^�)�  �          @�33�a�@�����\  C
=�a�@s33���H�,(�C�
                                    Bx^�8B  �          @����>�R?�
=��Q��u=qC���>�R@g���(��C��C�                                     Bx^�F�  �          @��#33?��R�У�G�C���#33@a���{�RG�B��R                                    Bx^�U�  �          @���R@���=q�{��C���R@��������AG�B�L�                                    Bx^�d4  �          @�R� ��@
=��Q��t��C� ��@��������8ffB��f                                    Bx^�r�  �          @����ff@1G����'  C#���ff@�z��aG����\C	�                                    Bx^���  �          @�z��tz�@s�
���H�!p�C)�tz�@�33�E����B��q                                    Bx^��&  �          @����
@G���\)��\CY����
@��H�>{��  C
Y�                                    Bx^���  �          @�\)��33@>�R�����33C����33@��<(���=qC�R                                    Bx^��r  �          @�G���{@O\)�}p���C����{@��
�+���Q�C�                                    Bx^��  �          @�G����\@\���z�H�Q�CǮ���\@����$z����C
�                                    Bx^�ʾ  �          @�
=���
@c33�g����CE���
@�G��\)��33C
h�                                    Bx^��d  �          @�\)��\)@hQ��Y�����C0���\)@���� �����C
�                                    Bx^��
  �          @�\���\@����$z����RC@ ���\@�z῁G���C	}q                                    Bx^���  �          @�\��=q@�33�!���C���=q@�ff�s33��p�C	\                                    Bx^�V  �          @������@�\)�+���G�C:�����@�(�������RC	.                                    Bx^��  �          @����p�@�G����(�C
�3��p�@�녿0����\)C�)                                    Bx^�"�  �          @����@�p������L  C�\���@�p�>�?��CaH                                    Bx^�1H  �          @��
��@�p���(��>�RC���@��
>�{@.�RC�                                    Bx^�?�  �          @����@��\�#�
��\)B������@�  ?�Q�A}B���                                    Bx^�N�  �          @�(����@��\��z��33C W
���@��?���AO�C�                                     Bx^�]:  �          @�ff���R@Z=q�L(�����C
=���R@��ÿ�{�p(�C�                                    Bx^�k�  �          @�����\@��e��z�Cc����\@_\)�#�
��
=C
                                    Bx^�z�  �          @��
��
=@���xQ��
=C����
=@\(��8�����C�)                                    Bx^��,  �          @�  ���H@Q��vff�p�C����H@a��4z���=qC�                                     Bx^���  �          @�p�����@*�H�n{���C8R����@qG��&ff���
C��                                    Bx^��x  �          @��H��ff@5��a���
=C�\��ff@w
=�
=���C��                                    Bx^��  �          @�Q���p�@33�XQ���\)C����p�@S�
�������C��                                    Bx^���  �          @߮��@
=q�Z=q���C.��@L(���R����C�                                    Bx^��j  �          @�
=��?�z��?\)��33C"!H��@3�
�
=q��  C�)                                    Bx^��  �          @�  ��=q@z��#33����C
��=q@C33��{�Up�CQ�                                    Bx^��  �          @�����
=@Q��4z���\)CG���
=@L�Ϳ��r�HC�
                                    Bx^��\  �          @����G�@(���Y����C����G�@h����\��ffC��                                    Bx^�  �          @������@,(��@  �ɅC{����@c33��33�z{Cs3                                    Bx^��  �          @������@*�H�G
=��G�C�q���@dz��   ��\)C��                                    Bx^�*N  �          @ᙚ��@8���8Q��\C���@l�Ϳ��H�c
=C��                                    Bx^�8�  �          @�(���{@,(��j�H��33C�q��{@q��!G�����C5�                                    Bx^�G�  �          @��H����@fff�,(���{CT{����@�녿���(��Cs3                                    Bx^�V@  �          @�\����@5�Vff��RC�
����@tz��
�H���C^�                                    Bx^�d�  �          @�(���@G��q�� =qC h���@L���6ff���\C�
                                    Bx^�s�  �          @�ff��
=?�=q��
=�C)Y���
=@{�e��p�C��                                    Bx^��2  �          @����G�?�G����\�
�HC*{��G�@
=�^{��=qC��                                    Bx^���  �          @�������?�  �y����C*B�����@�\�S�
���
C}q                                    Bx^��~  �          @�(����
@���>\@=p�Cc����
@���@�A��
C��                                    Bx^��$  �          @�z�����@�p�<#�
=��
C������@��H?�Ab�\C	}q                                    Bx^���  �          @�(����@��
�L�;�
=C����@��?�p�AYp�C	�\                                    Bx^��p  �          @��
���
@�Q쾊=q��C�R���
@�G�?��RA;�C
B�                                    Bx^��  �          @��H���@�Q콸Q�333C�3���@�
=?�AR{C
�)                                    Bx^��  �          @�\��33@�  >�  ?�(�C�3��33@�33?��RA{�CY�                                    Bx^��b  �          @�\����@�G�?�@�G�Cs3����@�G�@  A�
=Cn                                    Bx^�  �          @�=q���@���?O\)@˅C5����@�p�@!G�A��Cٚ                                    Bx^��  �          @�z����H@�=q>�\)@	��C :����H@�33@�
A�
=C�
                                    Bx^�#T  �          @�ff��z�@�33?:�H@��HCٚ��z�@�  @#�
A���C	@                                     Bx^�1�  �          @�p���(�@���?B�\@��Cn��(�@��@*=qA�{C�
                                    Bx^�@�  �          @�ff����@�{?�R@�G�C����@��
@   A�G�C�                                    Bx^�OF  �          @�
=��\)@���>�ff@_\)C  ��\)@���@Q�A���C�\                                    Bx^�]�  T          @�������@�ff?z�@���C8R����@�(�@#�
A��\C=q                                    Bx^�l�  �          @�����33@��?G�@���C �3��33@��H@0��A�\)C
                                    Bx^�{8  �          @�����@�(�?��A�B�
=����@��
@E�A�{Ch�                                    Bx^���  T          @���z�@�{?��HA��B�
=��z�@��
@O\)A�33C�f                                    Bx^���  �          @�=q��(�@��?�Q�A(�B��q��(�@��@QG�A�z�C�f                                    Bx^��*  �          @����\)@��H?�
=A�B��3��\)@���@QG�Aʣ�CaH                                    Bx^���  �          @��R���@�G�?�\)A$(�B�����@�z�@a�A�  C 33                                    Bx^��v  �          @�����=q@�ff?��A733B� ��=q@�
=@p  A�ffB�G�                                    Bx^��  �          @�
=�x��@�\)?���A;�
B���x��@�\)@r�\A�
=B���                                    Bx^���  �          @�\)�{�@�
=?��
A6�RB�=�{�@��@p  A�=qB�.                                    Bx^��h  �          @�����@��?�ffAX��B�33����@��
@s33A�C�                                    Bx^��  �          @�����@�(�@ ��As�
B��R���@�Q�@\)A�33CO\                                    Bx^��  �          @�ff���@�G�@A|  C �����@���@���A�
=C�                                     Bx^�Z  �          @�{���\@�(�@�AuG�C�H���\@���@{�A��C��                                    Bx^�+   �          @�{��{@���?�z�Ae�C���{@�ff@w
=A�C��                                    Bx^�9�  �          @������@��@   Av=qC�����@���@vffA�p�C	h�                                    Bx^�HL  �          @�����  @)��@ffA��\C{��  ?�\)@I��A���C"\                                    Bx^�V�  �          @ۅ��p�?���@Y��A�z�C#�
��p�>�  @p  B
��C1O\                                    Bx^�e�  �          @��H���
@��R?�z�A|Q�C�R���
@l(�@_\)A��C�{                                    Bx^�t>  �          @��H��{@��@!�A��C)��{@j=q@�z�B�C��                                    Bx^���  �          @�=q���@�Q�@,��A�33CY����@\��@��B\)Cu�                                    Bx^���  L          @�=q��{@���@'
=A��C{��{@B�\@\)B

=Ch�                                    Bx^��0  �          @��H��z�@���?�33A}��C	���z�@b�\@Z�HA�z�C�                                    Bx^���  �          @�33���R@|(�@-p�A�=qC����R@+�@}p�B=qC��                                    Bx^��|  �          @��H���@�@�B��C  ���>��
@�z�B3G�C0L�                                    Bx^��"  �          @�G���{?(��@��HBSG�C+���{��
=@�{BK�\CF޸                                    Bx^���  �          @�{�~�R@�@�Q�BF
=CW
�~�R<#�
@��B[�\C3�
                                    Bx^��n  �          @��
���H@�H@��B7�Ck����H>�ff@�ffBS�
C-�                                     Bx^��  �          @أ��n�R@tz�@�(�B�HCW
�n�R@�\@�  BK�HCh�                                    Bx^��  �          @���l��@?\)@��B5G�C\�l��?�G�@�=qB]�
C$��                                    Bx^�`  �          @���l��@��@��BAffC}q�l��>�G�@�G�B`Q�C-5�                                    Bx^�$  �          @�����(�@/\)@���B)�HCn��(�?^�R@�
=BLQ�C(�                                    Bx^�2�  "          @ڏ\�G
=@c�
@���B833C#��G
=?�  @��
Bn(�CE                                    Bx^�AR  �          @ٙ��AG�@}p�@�33B*�HB��3�AG�?�p�@��Bg=qC�                                    Bx^�O�  T          @ٙ��O\)@l(�@�B((�C=q�O\)?��@�\)B_�HC�                                    Bx^�^�  �          @�  �e@;�@��\B8��C�
�e?n{@�=qBa\)C%��                                    Bx^�mD  �          @�Q���@6ff@��B&\)C� ��?z�H@�\)BJQ�C&�{                                    Bx^�{�  "          @أ��{�@XQ�@�33B �CG��{�?\@���BN
=C�\                                    Bx^���  T          @�Q��Fff@W
=@�  B>�C�R�Fff?�G�@�z�Bqz�C�H                                    Bx^��6  
(          @�p��P��@P��@���B8�\C
=�P��?��R@��Biz�C8R                                    Bx^���  �          @�G��\)@�{@\��A�Q�C�)�\)@&ff@�Q�B3�C��                                    Bx^���  �          @�z����R@��R@*�HA��HC����R@XQ�@�\)B\)C=q                                    Bx^��(  
�          @�����@��\@*=qA�p�C\��@P  @�B
=C��                                    Bx^���  T          @�����@�p�@�A��C�q����@l��@�  B�C+�                                    Bx^��t  �          @�
=���@��?�A_
=Cz����@��@]p�A��
C��                                    Bx^��  �          @�ff��{@�33@�RA���CxR��{@j�H@xQ�B��CxR                                    Bx^���  �          @�ff��
=@����\)��\Cc���
=@�\)?�\)AYC�                                    Bx^�f  
�          @����ff@�p��G���p�C�H��ff@�
=�
=q��Q�CW
                                    Bx^�  �          @�p��Fff@��
���R�%Q�B���Fff@�=q�.{���B�\                                    Bx^�+�  �          @�  �z�@����=q���Bި��z�@љ�� ����(�B�(�                                    Bx^�:X  
�          @���\)@�����
=�-�B�(��\)@ə��3�
��  B��                                    Bx^�H�  �          @�����@|(�����KffB�����@����p�����B�                                    Bx^�W�  T          @�\�4z�@hQ����R�L�RB����4z�@����z�H�=qB��                                    Bx^�fJ  	�          @�G��5�@�\)����4�B����5�@��H�HQ���(�B�                                    Bx^�t�  "          @�R�Vff@�z��QG����B���Vff@�\)�z�H��B�                                    Bx^���  �          @���  @�ff����p�B�(���  @�Q�?�@���B�                                    Bx^��<  T          @���7�@��\�qG���33B��7�@˅��(��=�B��                                    Bx^���  �          @�{��
@���� �����\B���
@��H>�
=@g
=B�                                      Bx^���  "          @�p���\@��
�7
=��
=B�����\@�  ���
�$z�B�p�                                    Bx^��.  �          @�
=�,��@������H�
Q�B�8R�,��@�{��\�d(�B�\                                    Bx^���  �          @��<(�@��R����Q�B�k��<(�@Ǯ�����  B�                                     Bx^��z  �          @�\�E@�\)?(��@��RB�Ǯ�E@�Q�@?\)Aʏ\B                                    Bx^��   �          @�p��P��@��
>�@j�HB�(��P��@��R@6ffA���B�aH                                    Bx^���  
�          @�
=�B�\@�
=?O\)@���B�\�B�\@�{@G�AԸRB�W
                                    Bx^�l  "          @�Q��E�@�Q�?��@�Q�B�k��E�@��\@9��A��
B��)                                    Bx^�  �          @���\@أ�=L��>�(�BՅ��\@�\)@(��A�33B�33                                    Bx^�$�  �          @����p�@���?
=q@�{B��q��p�@�p�@'�A�
=C:�                                    Bx^�3^  T          @�{�w
=@��>�(�@fffB��f�w
=@�(�@&ffA�=qB��                                    Bx^�B  
�          @����aG�@��\�O\)��=qB�=q�aG�@�ff?�p�AG�
B�ff                                    Bx^�P�  "          @�  �j�H@��;��~{B�q�j�H@�(�?���Ar=qB�#�                                    Bx^�_P  "          @���R�\@��ÿ����.ffB�.�R�\@\?���A(�B���                                    Bx^�m�  �          @�=q�aG�@��׿��}�B��aG�@�=q>�ff@hQ�B�=q                                    Bx^�|�  
�          @�\)�n�R@�ff��z��T��B���n�R@�(�?B�\@���B�{                                    Bx^��B  "          @����z=q@��R�
�H��p�B�Ǯ�z=q@�33>B�\?��B�=q                                    Bx^���  
�          @�G���\)@����/\)���\C ����\)@�p��   �y��B�\                                    Bx^���  T          @����G�@����Fff��
=Cs3��G�@��H�k���B���                                    Bx^��4  "          @�R�e�@����N�R��{B�  �e�@�z�n{��{B�u�                                    Bx^���  T          @���N�R@�Q����{B����N�R@Ǯ���tQ�B�Ǯ                                    Bx^�Ԁ  
Z          @�{����@=p����
�"  Cٚ����@����G����
C
                                    Bx^��&  T          @����Mp�@�
=���
�G�B�aH�Mp�@�ff��(���  B��                                    Bx^���  T          @�\�Q�@�  � ����B�p��Q�@ȣ׽��
�&ffB�B�                                    Bx^� r  �          @���{@Ǯ�<(���ffB�  ��{@�zᾔz��Q�B�\)                                    Bx^�  "          @ᙚ�1G�@���5��ffB��f�1G�@θR��Q��;�B�aH                                    Bx^��  �          @��H���@�33�*�H��p�B޽q���@����Q�=p�B�p�                                    Bx^�,d  T          @���@ȣ��!G���{B�aH��@�Q�=�?uB���                                    Bx^�;
  T          @�33�@�ff����ffB���@�Q�?
=@���B�k�                                    Bx^�I�  �          @�\�p�@ʏ\�  ��(�B�k��p�@ָR>���@Mp�B�k�                                    Bx^�XV  "          @�\��\@�Q���m�Bֽq��\@�ff?^�R@�=qB���                                    Bx^�f�  "          @�\��=q@љ��Q���Q�B�#׿�=q@ۅ?
=@��B���                                    Bx^�u�  "          @�녿���@��
�����\)B�G�����@�=q>k�?�B�z�                                    Bx^��H  �          @�\����@�p���p��b�RBΏ\����@�=q?�G�A�
B�                                      Bx^���  
�          @��
�˅@��ÿ�(��>�\B�W
�˅@�=q?�ffA((�B�8R                                    Bx^���  �          @��H��p�@�  ��=q�,��B����p�@�\)?�A8Q�B��f                                    Bx^��:  �          @�� ��@׮�xQ����B�L�� ��@ҏ\?޸RAc�B�
=                                    Bx^���  �          @�
=�"�\@�(��h�����B�ff�"�\@�\)?�z�A^{B�\)                                    Bx^�͆  
�          @�녾��H@�?#�
@�G�B�����H@�(�@S�
A�z�B�(�                                    Bx^��,  �          @�׾.{@�  ?��HAC33B��)�.{@�z�@\)B(�B�k�                                    Bx^���  S          @��
�p��@`  �����1��C	)�p��@�{�P  ��G�B��H                                    Bx^��x  
�          @�\)���@�����(��\)C=q���@��\�
=��33C �
                                    Bx^�  T          @�\)���\@|(�������CǮ���\@����R����C��                                    Bx^��  �          @�
=���\@l(����\���C�����\@��H�!����RCxR                                    Bx^�%j  T          @�����@I�����R�z�C����@�Q��E��G�C\)                                    Bx^�4  T          @�ff����@aG�����\C@ ����@��\�:=q��33C�{                                    Bx^�B�  T          @��|��@�=q��z��#  C!H�|��@�p��8����p�B��q                                    Bx^�Q\  
�          @������@dz�����*33C�����@����R�\��(�C �
                                    Bx^�`  �          @�\)����@`  ��Q��(��C�)����@�{�N�R�̣�C�)                                    Bx^�n�  �          @���o\)@L����\)�@=qCk��o\)@�33�qG���(�B�k�                                    Bx^�}N  T          @����6ff@6ff����c{C  �6ff@��������HB��H                                    Bx^���  �          @�(��H��@HQ������Q��C#��H��@��������RB�q                                    Bx^���  
�          @��
�aG�@����(��  B�aH�aG�@�������r{B�W
                                    Bx^��@  
�          @�{����@�����Q���HC5�����@��׿���j�HB��                                    Bx^���  �          @�ff���@�����Q��z�C�f���@�\)�
�H��33B���                                    Bx^�ƌ  "          @�
=�vff@��
�vff��  B���vff@�\)��  �9B�                                     Bx^��2  T          @���{@�G��.�R����C���{@��R�   �x��Ch�                                    Bx^���  
�          @�R���R@��R�����=qCh����R@��?���A:�\CQ�                                    Bx^��~  �          @�z����\@�G�@  A�  CE���\@o\)@�=qB

=C�{                                    Bx^�$  
�          @�G���p�@��?�33A33Cs3��p�@�\)@EA���C	ٚ                                    Bx^��  �          @�����  @��?�ffA	�C8R��  @�
=@>�RAȸRC
c�                                    Bx^�p  "          @�\)���@�(�?�\@��RC+����@���@=qA�z�C
��                                    Bx^�-  �          @�ff��G�@��?�R@��C	5���G�@���@�A��HCT{                                    Bx^�;�  �          @ۅ���H@�ff?�=qA�C�\���H@xQ�@7
=A�{C=q                                    Bx^�Jb  
�          @�p���  @�
=?8Q�@��C����  @���@(��A��
C
�                                    Bx^�Y  T          @�33��p�@�{>��R@)��Ch���p�@�p�@
=qA��C�R                                    Bx^�g�  T          @�33��=q@�?�p�A&=qC�\��=q@s33@@  A�Q�C�R                                    Bx^�vT  �          @�{����@��?��RA$��C�\����@}p�@E�A���Cc�                                    Bx^���  T          @�
=��Q�@�p�?�(�A!C
��Q�@�  @L(�AٮC�                                    Bx^���  �          @�p�����@�p�?:�H@�=qC:�����@�ff@/\)A��\C�=                                    Bx^��F  �          @�\)��Q�@���>\)?�z�Cc���Q�@��@  A���C33                                    Bx^���  T          @�
=��=q@�Q�>�@r�\C�)��=q@��@�A��C	�                                    Bx^���  �          @��
��ff@�G�?˅AfffC�=��ff@aG�@QG�A���C�                                    Bx^��8  �          @�z��~{@e��ff�%  C	�f�~{@�p��7���B�{                                    Bx^���  
�          @�z���\)@_\)�����
C}q��\)@����1���G�C�                                    Bx^��  �          @�p���\)@i����  �p�C:���\)@�z��*=q��=qCk�                                    Bx^��*              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�v              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�&              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�4�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�Ch              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�R              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�`�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�oZ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�~               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^��L              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^��>              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^��0              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�|              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�"              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�-�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�<n              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�K              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�Y�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx^�h`   2          @���  @�{��p���
B�����  @���>Ǯ@H��B��                                    Bx^�w  I          @�
=�w
=@�33�g
=��C z��w
=@��Ϳ�=q�+\)B�aH                                    Bx^���  	.          @�\)�Z�H@���S33��z�B����Z�H@��ÿTz���(�B�#�                                    Bx^��R  	�          @�R�L��@��b�\���B���L��@��Ϳ���  B�{                                    Bx^���  �          @���HQ�@�33�����ffB����HQ�@ƸR��
��  B�ff                                    Bx^���  
�          @���3�
@���p��7{B����3�
@���A����HB�Ǯ                                    Bx^��D  
�          @�  � ��@`  ����RB�W
� ��@�  �p  ����B�.                                    Bx^���  	�          @�\)�P  @i�������<�
C���P  @�\)�Vff�ݮB�L�                                    Bx^�ݐ  
�          @�  �|(�@�G�������B��)�|(�@���=��
?(��B�ff                                    Bx^��6  �          @�G�����@�=q�p����HC � ����@����Q�@  B�#�                                    Bx^���  �          @�
=���@�p���{�.=qC{���@�Q�?s33@�33C��                                    Bx^�	�  �          @���33@��R�\�B�\B�\)��33@��H?n{@�ffB�33                                    Bx^�(  �          @�Q��^�R@Å��G��`z�B�aH�^�R@�G�?^�R@�B��f                                    Bx^�&�  �          @�=q�u�@ƸR����j�HB�Q��u�@��
@A�33B�G�                                    Bx^�5t  T          @�\�r�\@ȣ�=�?p��B�G��r�\@�@*=qA�{B�ff                                    Bx^�D  �          @��}p�@�>�ff@_\)B�L��}p�@�
=@:�HA��B��)                                    Bx^�R�  "          @�33�z�H@�z�?k�@�{B���z�H@��@UA�{B���                                    Bx^�af  �          @�(��g
=@�G�?��A'\)B�g
=@��R@q�A��RB�k�                                    Bx^�p  �          @���a�@˅?�\)A*=qB���a�@�Q�@u�A��B���                                    Bx^�~�  �          @�(��Z�H@�{@   A}�B��H�Z�H@��H@�33B  B��=                                    Bx^��X  
�          @�
=�K�@��
@*=qA�{B��
�K�@���@�=qB(�B�(�                                    Bx^���  �          @��H���@�(���
=�]p�C	�=���@���?�  AH��C=q                                    Bx^���  �          @������H?������'�C'p����H@<(��s�
���C�q                                    Bx^��J  �          @����@��@�p�?��A$��B�G��@��@��H@r�\A��B��
                                    Bx^���  "          @����	��@�{@W�A�B�8R�	��@�G�@�33BE�RB�.                                    Bx^�֖  T          @�{�H��@���?У�AV�HB���H��@��@|��B	�B�                                    Bx^��<  �          @�\)��ff?�p���G��=\)C{��ff@}p������=qC
c�                                    Bx^���  T          @�\)��z�@HQ�������Cc���z�@���*=q��z�C�3                                    Bx^��  
�          @�=q��{@��
���H�{33C:���{@�G��#�
��G�C
�f                                    Bx^�.  
(          @��
���H@_\)�Q���p�C  ���H@��׿����EG�C�                                    Bx^��  
�          @��
��(�@i���\(���33C޸��(�@�\)��z��Pz�C
��                                    Bx^�.z  �          @����@h���Z=q��(�C�\���@�
=����N=qC
��                                    Bx^�=   	.          @��
���@��H��H��C�)���@��;u���C��                                    Bx^�K�  
�          @�33����@��R�"�\��  C޸����@��H�����I��CJ=                                    Bx^�Zl  �          @�=q��
=@��R�����w�
C���
=@�=q>u?�33C�                                    Bx^�i  �          @������@��R�=q���C�����@�Q�8Q쿵C�                                    Bx^�w�  �          @�\��  @��H�.{����Cu���  @��ÿ�\����C��                                    Bx^��^  "          @�G���{@��\�
=q��{C�=��{@�  >L��?˅B�aH                                    Bx^��  
�          @�33���@����h����{C����@��
���R�:�HC �H                                    Bx^���  �          @���33@����ff���RCn��33@�=q�W
=���CT{                                    Bx^��P  T          @�z����\@�G��n{��Q�C�
���\@��R?�ffA"{CE                                    Bx^���  
�          @����G�@�Q��(��Z=qC�=��G�@�\)?޸RA]G�CY�                                    Bx^�Ϝ  �          @�\)��=q@��H>���@I��C
=��=q@�
=@"�\A��HC�
                                    Bx^��B  
�          @�z�����@�(��L�;ǮC������@�
=?�Q�A�z�C
p�                                    Bx^���  "          @����p�@�Q��������C����p�@�Q�aG����
C	\)                                    Bx^���  �          @�����ff@z=q?�{AX  C^���ff@:=q@C�
A���C�H                                    Bx^�
4  
�          @�=q���R@��\�5���HC�{���R@�?�A<z�C�3                                    Bx^��  
�          @�=q��ff@�p�����33C

��ff@�\)?�A:ffCB�                                    Bx^�'�  
Z          @�G�����@}p�>��H@��HC�����@Z�H@�
A��HC�\                                    Bx^�6&  T          @��H���@�G��s33����CǮ���@�
=?��
A'�C(�                                    Bx^�D�  �          @�(�����@��Ϳ޸R�j�RC����@��R>���@\)C�                                    Bx^�Sr  
�          @������@��H��H����C�����@��;u��z�Cu�                                    Bx^�b  	`          @�\)��ff@���>Ǯ@P��C���ff@�ff@33A�  C��                                    Bx^�p�  �          @޸R�p��@j�H@�p�B �C�R�p��?Ǯ@���BVz�C�=                                    Bx^�d  �          @������R@{�@G
=A�33C#����R@�@�{B �C��                                    Bx^��
  
�          @�p����\@��@FffA�
=C(����\@9��@�B#Q�C�f                                    Bx^���  
�          @�ff��@��@.�RA�ffC�)��@N{@�p�B�Cu�                                    Bx^��V  
�          @�{��z�@��@\)A��C����z�@i��@��\B
�\C��                                    Bx^���  "          @�\��  @��\@!�A��
C��  @J=q@�B�
CaH                                    Bx^�Ȣ  !          @�=q��@�\)@  A�33C	�q��@K�@xQ�B�
C0�                                    Bx^��H  �          @�33��=q@�Q�@�
A�\)CT{��=q@`��@tz�B�C�f                                    Bx^���  T          @�33��z�@��?��AH  C޸��z�@w
=@Y��A���C�                                    Bx^���  �          @���  @���?aG�@ᙚC�3��  @�Q�@6ffA�  C��                                    Bx^�:  
�          @�33���@�=q?�G�AfffC
(����@]p�@^{A�\C�)                                    Bx^��  
�          @��
��G�@���?��A(z�CxR��G�@g
=@AG�A�  C��                                    Bx^� �  �          @����H@���?8Q�@�33Cs3���H@hQ�@=qA��RC��                                    Bx^�/,  T          @����{@|��>�
=@X��C^���{@\(�?��RA�p�C�                                    Bx^�=�  T          @����p�@o\)?�  A�\C���p�@@��@(�A�G�C�                                    Bx^�Lx  "          @���\)@�  @   A�p�C�)��\)@)��@z=qBC=q                                    Bx^�[  
�          @�=q��
=@�(�@AG�A��C�f��
=@#33@�{Bp�C�R                                    Bx^�i�  
�          @���G�?�@��
BG�C ���G����
@���B(=qC4.                                    Bx^�xj  "          @������@\)@���B�\C������>��
@��B)�C0�=                                    Bx^��  �          @�z����@1�@��\B  C����?G�@��\B6�C*�3                                    Bx^���  �          @�\�q�@0  @���BA��C���q�>\@�BeffC.E                                    Bx^��\  �          @�Q����@s�
@��B
=C޸���?�G�@�p�BJ�HC��                                    Bx^��  
�          @��H��
=@�p�@dz�A�ffC����
=@%@��B6ffCff                                    Bx^���  �          @��
���@�=q@E�A�33C�����@;�@�p�B$
=C�H                                    Bx^��N  �          @��H��ff@�
=@\)A�(�C	���ff@K�@w
=B�HCE                                    Bx^���  �          @��
��33@�ff?���A\)C!H��33@W
=@333A��C�                                    Bx^��  �          @�(���
=@��
=��  C����
=@��?�z�A��C�
                                    